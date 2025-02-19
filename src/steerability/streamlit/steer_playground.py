import os
import re
import sys
import subprocess
import time

import numpy as np
import openai
from openai import OpenAI
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from goals import Goalspace
from llm_interactor import renormalize_goalspace
from utils.model_output_cleaner import clean_model_output

@st.cache_data
def load_probe():
    return pd.read_csv("./data/steerbench_converted.csv", index_col=0)

@st.cache_data
def load_seed_data():
    return pd.read_csv("./data/default_seed_data_goalspace_mapped.csv", index_col=0)

@st.cache_data
def load_goalspace():
    return Goalspace.create_default_goalspace_from_probe(st.session_state["probe"])

@st.cache_data
def parse_vllm_process(line):
    """Extract port, model name, API key, and host from a vllm process line."""
    if "vllm" not in line:
        return None  # Ensure we only process lines where 'vllm' is running
    
    match = re.search(
        r"vllm serve\s+([\w\-/]+)"
        r".*?--port (\d+)"
        r".*?--api-key (\S+)" 
        r".*?--host (\S+)", 
        line
    )
    
    if match:
        model_name, port, api_key, host = match.groups()
        return {"port": port, "model_name": model_name, "api_key": api_key, "host": host}
    
    return None 

@st.cache_data
def get_status():
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "vllm serve" in line:
            status = parse_vllm_process(line)
    model_name, host, port = status["model_name"], status["host"], status["port"]
    return f"**Currently running:** `{model_name}` at `http://{host}:{port}`" if status else "No VLLM instance is running.", status

@st.cache_data
def process_result(user_input, output_text, model_name):
    cleaned_output = clean_model_output(model_name, output_text)
    goalspace_out = st.session_state["goalspace"]([user_input, cleaned_output], return_pandas=True).add_prefix("output_raw_")
    out_normed = renormalize_goalspace(st.session_state["seed_data"], goalspace_out)
    goal_t = goalspace_out.T
    out_t = out_normed.T
    out_t.columns = ["source", "output"]
    out_t["diff"] = out_t.iloc[:, 1] - out_t.iloc[:, 0]
    return goal_t, out_t

def process_result_and_update(*args):
    goal_t, out_t = process_result(*args)
    out_t.index = [idx.split("_", 1)[-1] for idx in out_t.index]
    st.session_state["source"] = out_t["source"].to_dict()
    st.session_state["output"] = out_t["output"].to_dict()
    print(out_t)
    print("Updated session state:", st.session_state["source"])
    return goal_t, out_t

@st.cache_data(persist=True, show_spinner=False)
def fetch_cached_response(model, message):
    cached_response = generate_output(model, message)
    output_text = st.write_stream(cached_response)
    return output_text

def generate_output(model, message):
    try:
        completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": message,
                    }
                ],
                temperature=0, 
                stream=True,
            )
        return completion
    except openai.APIConnectionError:
        print("Raised a connection error -- check the endpoint.")

def color_map(val):
    """Apply color styling based on value."""
    if val == 0:
        return "background-color: white; color: black"
    elif val > 0:
        red_intensity = int(255 * val)  # Scale red from 0 to 255
        return f"background-color: rgb({red_intensity}, 0, 0); color: white"
    else:
        blue_intensity = int(255 * abs(val))  # Scale blue from 0 to 255
        return f"background-color: rgb(0, 0, {blue_intensity}); color: white"
    
def get_badness_color(value, bad_threshold):
    if np.isnan(value):
        return "rgb(255,255,255)" # white
    norm_value = min(value / bad_threshold, 1.0)  # Normalize value to [0, 1] range
    r = int(255 * norm_value)  # More red as value increases
    g = int(255 * (1 - norm_value))  # Less green as value increases
    return f"rgb({r},{g},0)" 

def scalar_rejection(a, b): # b onto a
    return np.sqrt(np.linalg.norm(b) ** 2 - (np.sum(a * b) / (np.linalg.norm(a) + 1e-8)) ** 2)

def scalar_projection(a, b):
    return np.sum(a * b) / np.linalg.norm(a)

st.title("Steerability Playground")

with st.spinner(text="Initializing goal-space..."):
    if "probe" not in st.session_state:
        st.session_state["probe"] = load_probe()
    if "goalspace" not in st.session_state:
        st.session_state["goalspace"] = load_goalspace()
    if "seed_data" not in st.session_state:
        st.session_state["seed_data"] = load_seed_data()

TOXICITY = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]
NON_TOXIC = [g for g in st.session_state["goalspace"].get_goal_names(snake_case=True) if g not in TOXICITY]

inst = st.text_area("Instruction:", height=100)
user_input = st.text_area("Source text:", height=200)
if "prompt" not in st.session_state:
    st.session_state["inst"] = inst
    st.session_state["prompt"] = user_input


for attr in ["deltas", "source", "output"]:
    if attr not in st.session_state:
        st.session_state[attr] = {goal: "0.0" for goal in st.session_state["goalspace"].get_goal_names(snake_case=True) if goal not in TOXICITY}

st.write("**Goal deltas**")
cols = st.columns([1] + [1.2] * len(NON_TOXIC))  
max_display_len = 20

for i, (goal, col) in enumerate(zip([""] + NON_TOXIC, cols)):
    if i == 0:
        with col:
            st.markdown("<div style='margin-top: 35px;'></div>", unsafe_allow_html=True)
            st.write("**Deltas**")
            st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
            st.write("**Source**")
            st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
            st.write("**LLM Response**")
    else:
        with col:
            st.session_state["deltas"][goal] = st.text_input(
                goal if len(goal) <= max_display_len else goal[:max_display_len - 3] + "...", st.session_state["deltas"][goal], key=f"input_{goal}", 
            )
            st.text_input(
                goal, st.session_state["source"][goal], disabled=True, key=f"source_{goal}", label_visibility="collapsed"
            )
            st.text_input(
                goal, st.session_state["output"][goal], disabled=True, key=f"output_{goal}", label_visibility="collapsed"
            )


st.divider()
st.write("### Steerability (given goals)")
df = pd.DataFrame([st.session_state["deltas"], st.session_state["source"], st.session_state["output"]], index=["deltas", "source", "output"]).astype(float)
df = df.loc[:, df.columns.isin(NON_TOXIC)]

error_vec = df.loc["source"] + df.loc["deltas"] - df.loc["output"] # z^* - \hat{z}
what_i_got = df.loc["output"] - df.loc["source"] # \hat{z} - z0
what_i_wanted = df.loc["deltas"] # z^* - z0

print("Vectors")
print(error_vec)
print(what_i_got)
print(what_i_wanted)
distance_to_target = np.linalg.norm(error_vec)

requested_mvmt = np.linalg.norm(what_i_wanted)
llm_movement = np.linalg.norm(what_i_got)

raw_ortho =  scalar_rejection(what_i_wanted, error_vec) 
ortho = raw_ortho / (llm_movement + 1e-8)

raw_miscal =  np.abs(scalar_projection(what_i_wanted, error_vec))
miscalibration = raw_miscal / (requested_mvmt + 1e-8)

col1, col2, col3 = st.columns(3)

comp_error = np.abs(np.sqrt(raw_miscal ** 2 + raw_ortho ** 2) - distance_to_target)

with col1:
    color = get_badness_color(distance_to_target, 1)
    st.markdown(f"<div style='text-align: center; font-size: 36px; font-weight: bold; color: {color};'>{distance_to_target:.3f}</div>", unsafe_allow_html=True)
    st.caption(f"Steering Error (decomp. error: {comp_error:.2E})")

with col2:
    color = get_badness_color(miscalibration, 0.5)
    st.markdown(f"<div style='text-align: center; font-size: 36px; font-weight: bold;  color: {color};'>{miscalibration:.3f}</div>", unsafe_allow_html=True)
    st.caption(f"Miscalibration (raw: {raw_miscal:.3f}; requested movement: {requested_mvmt:.3f})")

with col3:
    color = get_badness_color(ortho, 0.9)
    st.markdown(f"<div style='text-align: center; font-size: 36px; font-weight: bold; color: {color};'>{ortho:.3f}</div>", unsafe_allow_html=True)
    st.caption(f"Orthogonality (raw: {raw_ortho:.3f}; observed movement: {llm_movement:.3f})")

st.divider()
status_text, status_dict = get_status()
host, port = status_dict["host"], status_dict["port"]
client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key=status_dict["api_key"])
st.markdown(status_text)


output_text = None
if st.button("Send API Request"):
    start = time.time()
    with st.spinner("Running..."):
        output_placeholder = st.empty()
        
        output_text = fetch_cached_response(status_dict["model_name"], inst + "\n\n" + user_input)
        elapsed = time.time() - start
        st.session_state["last_response"] = output_text
        st.session_state["elapsed"] = elapsed

if "last_response" in st.session_state:
    if output_text is None:
        output_text = st.session_state["last_response"]
        elapsed = st.session_state.get("elapsed", 0)
        st.write(output_text)
    len_in, len_out = len(user_input.split()), len(output_text.split())
    st.divider()
    st.markdown("**API call stats:**")
    st.text(f"Prompt tokens: {len_in} ({len_in / elapsed:.2f} tok/s)")
    st.text(f"Output tokens: {len_out} ({len_out / elapsed:.2f} tok/s)")
    _, normalized = process_result_and_update(user_input, output_text, status_dict["model_name"])
    st.rerun()



    