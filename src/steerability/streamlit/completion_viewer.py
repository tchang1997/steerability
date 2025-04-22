import asyncio
from collections import defaultdict
import glob
import os

from aiohttp import ClientSession
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import textstat
from typing import Optional

from steerability.rewards import send_request

EVAL_OUTPUTS_DIR = os.path.join(
    os.path.expanduser("~"),
    "steerability/src/steerability/eval_outputs"
)
POSSIBLE_GOALS = ["reading_difficulty", "formality", "textual_diversity", "text_length"]
PORT = 16641

def display_metrics(orig, *completions):

    for col in st.columns(len(completions)):
        with col:
            st.markdown("### Metrics breakdown")

    cols = st.columns(len(completions) * 2)
    col_splits = [cols[2*i:2*i+2] for i in range(len(completions))]

    for i, col_split in enumerate(col_splits):
        completion = completions[i]
        with col_split[0]:
            st.markdown(f"**Flesch-Kincaid Score:** {textstat.flesch_kincaid_grade(completion):.2f} (orig.: {textstat.flesch_kincaid_grade(orig):.2f})")
            st.markdown(f"- avg. words/sent.: {textstat.words_per_sentence(completion):.2f} (orig.: {textstat.words_per_sentence(orig):.2f})")
            st.markdown(f"- avg. syll./word: {textstat.avg_syllables_per_word(completion):2f} (orig.: {textstat.avg_syllables_per_word(orig):.2f})")

        with col_split[1]:
            st.markdown(f"**Heylihgen-Dewaele Score:** {FORMALITY(completion):.2f} (orig.: {FORMALITY(orig):.2f})")
            freqs = FORMALITY.get_pos_freqs(completion)
            orig_freqs = FORMALITY.get_pos_freqs(orig)
            for pos, pct in freqs.items(): 
                st.markdown(f"- % {pos}: {pct:.2f}% (orig.: {orig_freqs[pos]:.2f}%)")
            diectic_coeff = freqs["NOUN"] + freqs["ADJ"] + freqs["ADP"] + freqs["ARTICLE"]
            non_diectic_coeff = freqs["PRON"] + freqs["VERB"] + freqs["ADV"] + freqs["INTJ"] 
            orig_diectic_coeff = orig_freqs["NOUN"] + orig_freqs["ADJ"] + orig_freqs["ADP"] + orig_freqs["ARTICLE"]
            orig_non_diectic_coeff = orig_freqs["PRON"] + orig_freqs["VERB"] + orig_freqs["ADV"] + orig_freqs["INTJ"] 

            st.markdown(f"*Total diectic (N+ADJ+ADP+ART)*: {diectic_coeff:.2f}% (orig.: {orig_diectic_coeff:.2f}%)")
            st.markdown(f"*Total non-diectic (PRON+V+ADV+INT)*: {non_diectic_coeff:.2f}% (orig.: {orig_non_diectic_coeff:.2f}%)")

    st.divider()

@st.cache_resource
def get_goalspace():
    try:
        from steerability.goals import DEFAULT_GOALS, Goalspace
    except ImportError as e:
        import sys
        raise ImportError(f"Unable to import steerability. sys.path is {sys.path}")
    form = Goalspace(DEFAULT_GOALS)
    return form        

@st.cache_resource
def get_formality_scorer():
    try:
        from steerability.goals import Formality
    except ImportError as e:
        import sys
        raise ImportError(f"Unable to import steerability. sys.path is {sys.path}")
    form = Formality()
    return form

FORMALITY = get_formality_scorer()

@st.cache_data
def load_dfs(run_dir):
    df_files = glob.glob(os.path.join(run_dir, "*.csv"))
    dfs = [pd.read_csv(f, index_col=0) for f in df_files]
    if len(dfs):
        df = pd.concat(dfs, ignore_index=True)
        return df
    
@st.cache_data
def load_eval(run_name):
    eval_file = os.path.join("./training_probes", run_name + "_eval.csv")
    eval_df = pd.read_csv(eval_file, index_col=0)
    return eval_df

async def safe_request(i, session, text, goals, port):
    try:
        if not isinstance(text, list):
            text = [text]
        result = await send_request(session, text, goals=goals, port=port, normalize=True)
        return i, result
    except Exception as e:
        print("An exception was raised when sending request", i)
        return i, e 

async def get_goalspace_mappings(responses: list[str], batch_size: Optional[int] = 4):
    goalspace = get_goalspace()
    goals = goalspace.get_goal_names(snake_case=True)
    resps_batched = [responses[i:i + batch_size] for i in range(0, len(responses), batch_size)]

    results = [None] * len(resps_batched)
    sem = asyncio.Semaphore(8)  

    async def limited_safe_request(i, session, text):
        async with sem:
            return await safe_request(i, session, text, goals=goals, port=PORT)

    async with ClientSession() as session:
        tasks = [limited_safe_request(i, session, text) for i, text in enumerate(resps_batched)]

        for fut in asyncio.as_completed(tasks):
            i, response = await fut
            results[i] = response # dict[str -> list[float]]

    # handle lists of lists in future
    combined = defaultdict(list)
    for d in results:
        for k, v in d.items(): # str -> list[float]
            combined[k].extend(v)
    mappings = pd.DataFrame(combined)
    return mappings

def rgba_colors(base_colors, alpha_fade):
    return [
        f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a:.3f})'
        for (r, g, b, _), a in zip([plt.cm.jet(c) for c in base_colors], alpha_fade)
    ]

@st.cache_data
def cache_mappings(response_list):
    return asyncio.run(get_goalspace_mappings(response_list))


def plot_tracks(eval_df, response_list, prompt_id):
    allowed_goals = [c for c in eval_df.columns if c in POSSIBLE_GOALS]
    x_axis = st.selectbox("X-axis", allowed_goals, index=0)
    y_axis = st.selectbox("Y-axis", allowed_goals, index=1)

    x_final = eval_df.iloc[prompt_id][f"target_{x_axis}"]
    x_init = eval_df.iloc[prompt_id][f"source_{x_axis}"]
    y_final = eval_df.iloc[prompt_id][f"target_{y_axis}"]
    y_init = eval_df.iloc[prompt_id][f"source_{y_axis}"]

    mappings = cache_mappings(response_list)
    x = mappings[x_axis]
    y = mappings[y_axis]

    # Optional smoothing
    alpha = 0.2
    x_smooth = pd.Series(x).ewm(alpha=alpha).mean().values # smooth outputs
    y_smooth = pd.Series(y).ewm(alpha=alpha).mean().values

    colors = np.linspace(0, 1, len(x))  # for colormap
    plotly_colors = [
        f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'
        for r, g, b, _ in [plt.cm.jet(c) for c in colors]
    ]

    frames = []
    for i in range(1, len(x)):
        fade_alphas = np.linspace(0.05, 0.8, i+1) ** 1.2
        faded_colors = rgba_colors(colors[:i+1], fade_alphas)

        frames.append(
            go.Frame(
                name=str(i),
                data=[
                    go.Scatter(
                        x=x[:i+1], y=y[:i+1],
                        mode="markers",
                        marker=dict(symbol="circle", size=8, color=faded_colors),
                        showlegend=False
                    ),
                    go.Scatter(
                        x=[x_init, x_final],
                        y=[y_init, y_final],
                        mode="lines",
                        line=dict(color="black", width=2),
                        showlegend=False
                    ),
                ],
                layout=go.Layout(
                    annotations=[
                        dict(
                            x=x_smooth[i], y=y_smooth[i],
                            ax=x_init, ay=y_init,
                            xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True, arrowhead=2, arrowsize=1,
                            arrowcolor=plotly_colors[i], opacity=1.0
                        ),
                        dict(
                            x=x_final, y=y_final,
                            ax=x_init, ay=y_init,
                            xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor="black",
                            opacity=0.8
                        )
                    ]
                )
            )
        )



    fig = go.Figure(
        data=[
            go.Scatter(
            x=[x[0]], y=[y[0]],
            mode="markers",
            marker=dict(symbol="circle", size=8, color=[rgba_colors([colors[0]], [0.8])[0]]),
            showlegend=False
            ),
            # Static black arrow
            go.Scatter(
                x=[x_init, x_final],
                y=[y_init, y_final],
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False
            ),
        ],
        frames=frames
    )



    # Layout (including initial shape)
    fig.update_layout(
        paper_bgcolor="black",
        plot_bgcolor="white",
        font_color="black",
        width=600,  # You can tweak this if needed
        height=600,
        xaxis=dict(
            title=x_axis,
            range=[-0.05, 1.05],
            showgrid=True,
            gridcolor="black",
            scaleanchor="y",  # <- force square aspect ratio
            scaleratio=1,
            constrain="domain",  # <- don't stretch x-axis
        ),
        yaxis=dict(
            title=y_axis,
            range=[-0.05, 1.05],
            showgrid=True,
            gridcolor="black",
            constrain="domain",  # <- don't stretch y-axis either
        ),
        margin=dict(t=40, r=20, b=40, l=50),
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Step: ",
                "visible": True,
                "xanchor": "right"
            },
            "pad": {"b": 10, "t": 50},
            "steps": [
                {
                    "method": "animate",
                    "label": str(i),
                    "args": [
                        [str(i)],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": True},
                            "transition": {"duration": 0}
                        }
                    ]
                }
                for i in range(1, len(x))
            ]
        }],
        annotations=[
            dict(
                x=x_final, y=y_final,
                ax=x_init, ay=y_init,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,         # pick arrowhead style
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="black",
                opacity=0.8
            )
        ]
    )

    fig.update_xaxes(title_text=x_axis)
    fig.update_yaxes(title_text=y_axis)
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def get_figure(metric_cols, prompt_id):
    fig, axs = plt.subplots(len(metric_cols), 1, figsize=(16, 5 * len(metric_cols)))

    for col, ax in zip(metric_cols, axs):
        # Plot all prompts in background
        for i in range(n_examples):
            df_sub = df_split.groupby("step").nth(i).sort_values(by="step")
        
            if i == prompt_id:
                continue  # plot the selected one separately
            ax.plot(
                df_sub["step"],
                df_sub[col],
                color="tab:blue",
                alpha=0.4,
                linewidth=1,
                marker=None
        )

        # Plot selected prompt on top
        ax.plot(
            df_prompt["step"],
            df_prompt[col],
            label=f"Selected prompt" if col == metric_cols[0] else None,
            color="blue",
            alpha=1.0,
            linewidth=1.5,
            marker='o',
            markersize=4
        )

        ax.plot(
            df_prompt["step"],
            df_split.groupby("step")[col].median(),
            color="black",
            alpha=1.0,
            linewidth=2.,
            linestyle="dashed",
            label="Mean" if col == metric_cols[0] else None,
        )

        ax.plot(
            df_prompt["step"],
            df_split.groupby("step")[col].quantile(0.75),
            color="black",
            alpha=0.8,
            linewidth=2.,
            linestyle="dotted",
            label="Q3" if col == metric_cols[0] else None
        )

        ax.plot(
            df_prompt["step"],
            df_split.groupby("step")[col].quantile(0.25),
            color="black",
            alpha=0.8,
            linewidth=2.,
            linestyle="dotted",
            label="Q1" if col == metric_cols[0] else None,
        )

        ax.set_title(col)
        ax.set_xlabel("Step")
        ax.set_ylabel(col)
        ax.grid(True)

    fig.legend(bbox_to_anchor=(0.5, -0.02), loc='lower center', ncols=4)
    fig.tight_layout()
    return fig

def show_metrics(df_prompt, prompt_id):
    st.markdown("### Metric Trajectories")
    # Grab relevant columns
    metric_cols = [col for col in df_prompt.columns if col.endswith("_wrapper") or (col.startswith("get_") and col.endswith("_err"))]
    if metric_cols:
        fig = get_figure(metric_cols, prompt_id)
        st.pyplot(fig)
        st.markdown("**Raw Data**")
        st.dataframe(df_prompt)
    else:
        st.warning("No matching metric columns found.")


st.title("Completion viewer")

# Step 1: Run selection
run_names = sorted(os.listdir(EVAL_OUTPUTS_DIR))
selected_run = st.selectbox("Select a run", run_names)

# Step 2: Load and concat dataframes
run_dir = os.path.join(EVAL_OUTPUTS_DIR, selected_run)
df = load_dfs(run_dir)
if st.button("Force reload", on_click=load_dfs.clear):
    load_dfs.clear()

if df is not None:
    # Step 3: Select split
    split = st.selectbox("Select split", df["split"].unique())

    # Step 4: Select prompt_id
    df_split = df[df["split"] == split]
    #unique_prompts = df_split["prompt"].unique()
    #prompt_indices = range(len(unique_prompts))
    first_group = df_split.loc[df_split["step"] == df_split["step"].min(), "prompt"]
    n_examples = len(first_group)
    prompt_id = st.selectbox("Select prompt", range(n_examples))
    df_prompt = df.groupby("step").nth(prompt_id).sort_values(by="step")


    # Step 5: Show prompt
    st.markdown("### Prompt")
    prompt_text = df_prompt.iloc[0]["prompt"]
    st.code(prompt_text, language="text", wrap_lines=True)

    # Step 6: Slider-based diff viewer
    
    steps = sorted(df_prompt["step"].unique())

    col1, col2 = st.columns(2)
    with col1:
        step1 = st.select_slider("Step (left)", options=steps, key="slider1")
        row1 = df_prompt[df_prompt["step"] == step1]
        st.markdown("**Output (left) **")
        st.code(row1["completion"].item(), language="text", wrap_lines=True)
        #st.text_area("Output (left)", row1["completion"], height=300)

    with col2:
        step2 = st.select_slider("Step (right)", options=steps, key="slider2")
        row2 = df_prompt[df_prompt["step"] == step2]
        st.markdown("**Output (right) **")
        st.code(row2["completion"].item(), language="text", wrap_lines=True)

    display_metrics(prompt_text, row1["completion"].item(), row2["completion"].item())


    st.markdown("### Steering plots")
    eval_df = load_eval(selected_run)
    plot_tracks(eval_df, df_prompt["completion"].tolist(), prompt_id)

    show_metrics(df_prompt, prompt_id)