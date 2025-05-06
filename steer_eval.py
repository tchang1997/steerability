from argparse import ArgumentParser
import json
import os
from pathlib import Path
import pickle
import requests
from string import Template as StringTemplate
import subprocess
import time

import pandas as pd
from ruamel.yaml import YAML

from steerability.llm_interactor import LLMInteractor
from steerability.llm_launcher import is_huggingface_model, is_openai_model, is_local_path, start_vllm_server
from steerability.instruction_generator import get_instruction_generator
from steerability.utils.llm_groundedness import create_problem_set, parse_and_score, interactive_review
from steerability.utils.pairwise_goal_validation import initialize_chat_instance, judge

from typing import Any

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

RESULT_DIR = Path("./results")
DEFAULT_LLM_CALL_TIMEOUT = 3600
GOALSPACE_START_TIMEOUT = 600
HEALTH_CHECK_FREQ = 10

def block_until_healthy(proc, port: int):
    for _ in range(GOALSPACE_START_TIMEOUT // HEALTH_CHECK_FREQ):
        try:
            resp = requests.get(f"http://localhost:{port}/health")
            if resp.status_code == 200:
                logger.info("✅ vLLM server ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(HEALTH_CHECK_FREQ)
    else:
        logger.error(f"❌ vLLM server did not start in time (timeout: {GOALSPACE_START_TIMEOUT}s).")
        proc.terminate()
        return False

def launch_goalspace_server(uvicorn_cfg: dict):
    env = os.environ.copy()
    env["GOAL_DIMENSIONS"] = ",".join(uvicorn_cfg["goals"])
    env["NORMALIZATION_PATH"] = uvicorn_cfg["normalization"]
    command = [
        "uvicorn",
        "steerability.goalspace_server:app",
        "--port", str(uvicorn_cfg["port"]),
        "--host", str(uvicorn_cfg["host"]),
        "--num-workers", str(uvicorn_cfg["num_workers"]),
    ]
    pid = os.getpid()
    with open(f"logs/{pid}-uvicorn.out", "w") as out, open(f"logs/{pid}-uvicorn.err", "w") as err:
        proc = subprocess.Popen(command, env=env, stdout=out, stderr=err)
    return proc

def get_nonnull_targets(probe: pd.DataFrame):
    target_goals = probe.filter(like='target_', axis=1)
    source_goals = probe.filter(like="source_", axis=1)

    target_corrected_goals = target_goals.copy()
    for target_col in target_goals.columns:
        source_col = target_col.replace("target_", "source_")
        target_corrected_goals[target_col] = target_goals[target_col].fillna(source_goals[source_col])
    return target_corrected_goals

def launch_steerability_eval(
        probe: pd.DataFrame,
        chat_type: str,
        cfg: dict[str, Any],
        vllm_cfg: dict[str, Any],
        uvicorn_cfg: dict[str, Any],
    ):
    prompt_strategy = cfg["prompt_strategy"]
    logger.info(f"Using prompt strategy `{prompt_strategy}`")

    database = None
    if "instruction_path" in cfg: 
        if Path(cfg["instruction_path"]).is_file():
            logger.info("Loading instruction database at %s", cfg["instruction_path"])
            with open(cfg["instruction_path"], "rb") as f:
                database = pickle.load(f)

    instruction_generator = get_instruction_generator(
        prompt_strategy,
        database=database, # extracted via LLM from CoT traces
        prompter_kwargs=cfg.get("prompter_kwargs", {})
    )

    llm_interactor = LLMInteractor(
        instruction_generator,
        cfg["model_id"],
        chat_type,
        api_config=args.api_config,
        cache_file=Path("./cache") / (cfg["model_id"] + ".tsv"),
        timeout=DEFAULT_LLM_CALL_TIMEOUT,
        async_mode=True,
        port=vllm_cfg["port"],
        goalspace_port=uvicorn_cfg["port"],
        max_tokens=cfg["max_tokens"],
        max_simul_goalspace_reqs=uvicorn_cfg["num_workers"],
        text_gen_kwargs=cfg["text_gen_kwargs"],
    )
    goalspace_proc = launch_goalspace_server(uvicorn_cfg) # non-blocking

    delta_goals = probe.filter(like='delta_', axis=1)
    target_corrected = get_nonnull_targets(probe)

    logger.info("Writing prompts...")
    prompts = llm_interactor.write_prompts(
        delta_goals,
        target_corrected
        **cfg.get("inst_addons", {})
    ) 
    logger.info("EXAMPLE PROMPT:")
    logger.info(f"---\n{prompts[0]}---\n") 
    logger.info("Pinging endpoint:", llm_interactor.url_endpoint)

    outputs = llm_interactor.generate_steerability_data(
        probe,
        prompts,
        normalization_cfg="./config/goalspace_defaults/normalization.yml", # one day let's not hardcode this
    ) # seed_data for normalization only -- refactor someday?
    goalspace_proc.terminate()
    return outputs

def launch_judge(judge_config: str):
    proc = start_vllm_server(judge_config["model"], judge_config["vllm_args"]) # blocks until healthy
    if proc is None:
        raise RuntimeError(f"vLLM server failed to start in time. Try pre-downloading the model weights, or file an issue.")

    return proc

def launch_llm(model_name: str, vllm_config: str):
    model_name = cfg["model_id"]
    if is_local_path(model_name) or is_huggingface_model(model_name):
        chat_type = "vllm"
        proc = start_vllm_server(model_name, vllm_config) # blocks until healthy
        if proc is None:
            raise RuntimeError(f"vLLM server failed to start in time. Try pre-downloading the model weights, or file an issue.")
    elif is_openai_model(model_name):
        chat_type = "openai"
    else:
        raise ValueError(f"Model '{model_name}' not recognized as a local path, HuggingFace model, or OpenAI model.")
    return proc, chat_type

def compute_steerability_metrics(probe: str):
    pass

if __name__ == '__main__': 
    psr = ArgumentParser()
    psr.add_argument("--config", type=str, required=True, help="Config file for steerability evaluation.")
    psr.add_argument("--api-config", type=str, help="JSON containing API key; e.g., {'api_key': 'sk-...'}")
    psr.add_argument("--vllm-config", type=str, default="./config/vllm_defaults/openai_server.yml", help="vLLM server config file.")
    psr.add_argument("--uvicorn-config", type=str, default="./config/uvicorn_defaults/goalspace_server.yml", help="Goalspace-mapping server config.")
    psr.add_argument("--skip-judge", action="store_true", help="Whether to skip the interactive LLM as judge phase.")
    psr.add_argument("--judge-model", type=str, default="meta-llama/Llama-3.1-Instruct-8B", help="Model ID for judge model.")
    psr.add_argument("--judge-config", type=str, default="./config/vllm_defaults/judge_model.yml", help="vLLM judge model config.")
    args = psr.parse_args()

    yaml = YAML(typ="safe")
    with open(args.config, "r") as f:
        cfg = yaml.load(f)
    with open(args.vllm_config, "r") as f:
        vllm_cfg = yaml.load(f)
    with open(args.uvicorn_config, "r") as f:
        uvicorn_cfg = yaml.load(f)

    name = cfg["save_as"]
    result_path = RESULT_DIR / "raw" / (name + ".csv")
    judged_path = RESULT_DIR / "judged" / (name + ".csv")
    final_metrics_path = RESULT_DIR / "steerability_metrics" / (name + ".json")

    probe = None
    probe_path = cfg["probe"]
    if not Path(probe_path).is_file():
        raise FileNotFoundError(f"Steerability probe not found at {probe_path}.")
    else:
        probe = pd.read_csv(probe_path, index_col=0, low_memory=False)

    proc, chat_type = launch_llm(cfg["model_id"], vllm_cfg)

    raw_probe = launch_steerability_eval(probe, chat_type, uvicorn_cfg) # incl. reprompt if 5XX
    raw_probe.to_csv(result_path)
    proc.terminate()

    if not args.skip_judge:
        with open(args.judge_config, "r") as f:
            judge_cfg = yaml.load(f)
        judge_proc = launch_judge(judge_cfg)
        with open(judge_cfg["prompt_file"], "r") as f:
            prompt_template = StringTemplate(f.read().strip())
        judge_port = judge_cfg["vllm_args"]["port"]
        pset = create_problem_set(
            probe,
            prompt_template,
            judge_cfg["source_col"],
            judge_cfg["response_col"],
            judge_cfg["seed"],
            port=judge_port,
        )
        chat_instance = initialize_chat_instance(
            judge_port,
            api_config=args.api_config,
            cache_suffix="_groundedness_judge.tsv"
        )
        pset_answered = judge(pset, chat_instance)
        judged = parse_and_score(pset_answered)

        reviewed = interactive_review(judged, judge_cfg["spot_check_size"])
        reviewed.to_csv(judged_path)

    raw_json = compute_steerability_metrics(raw_probe)
    with open(final_metrics_path) as f:
        json.dump(raw_json, f)
