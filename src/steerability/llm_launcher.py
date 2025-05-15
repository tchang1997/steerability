import os
from pathlib import Path
import requests
import subprocess
import time

import openai
from openai import OpenAI
from transformers import AutoConfig
from ruamel.yaml import YAML
yaml = YAML(typ="safe")

from typing import Optional

import logging
logger = logging.getLogger(__name__)

MAX_VLLM_START_TIMEOUT = 7200
HEALTH_CHECK_FREQ = 30

def get_max_model_len(config: AutoConfig, max_model_len: int) -> int:
    # from old version of vLLM
    possible_keys = [
        # OPT
        "max_position_embeddings",
        # GPT-2
        "n_positions",
        # MPT
        "max_seq_len",
        # ChatGLM2
        "seq_length",
        # Command-R
        "model_max_length",
        # Whisper
        "max_target_positions",
        # Others
        "max_sequence_length",
        "max_seq_length",
        "seq_len",
    ]
    for key in possible_keys:
        max_len_key = getattr(config, key, None)
        if max_len_key is not None:
            max_model_len = min(max_model_len, max_len_key)
    return max_model_len

def is_local_path(model_name: str) -> bool:
    return Path(model_name).exists()

def is_huggingface_model(model_name: str) -> bool:
    return "/" in model_name 

def is_openai_model(model_name: str) -> bool:
    env = os.environ
    if "OPENAI_API_KEY" not in env: 
        raise RuntimeError("OpenAI API key must be passed via environment variable. Set OPENAI_API_KEY='sk-...' or add to your .bashrc/.zshrc file.")
    client = OpenAI()
    try:
        _ = client.models.retrieve(model_name)
        return True
    except openai.NotFoundError:
        logger.error(f"Model '{model_name}' is not an OpenAI model.")
        return False

def start_vllm_server(model_name: str, cfg: str, is_judge: Optional[bool] = False):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # flush logs immediately
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
        logger.info(f"Setting CUDA visible devices for vLLM instance to {env['CUDA_VISIBLE_DEVICES']}")
    if is_local_path(model_name):
        env["USE_FASTSAFETENSOR"] = "true"
        logger.info(f"Local model path detected. Setting USE_FASTSAFETENSOR='true'.")

    config = AutoConfig.from_pretrained(model_name, cache_dir=os.environ.get("HF_HOME", "~/.cache/huggingface/hub"))
    derived_max_model_len = get_max_model_len(config, cfg["max_model_len"]) 
    if derived_max_model_len < cfg["max_model_len"]:
        logger.warning(f"Maximum context length of {model_name} is {derived_max_model_len} via config.json. "
            f"Setting --max-model-len to {derived_max_model_len}. This will cause errors downstream if your "
            f"config file requests completions that exceed the context length.")

    port = str(cfg["port"])
    vllm_cmd = [
        "vllm", "serve", model_name,
        "--host", "localhost",
        "--port", port,
        "--max-model-len", str(derived_max_model_len),
        "--gpu-memory-utilization", str(cfg["gpu_memory_utilization"]),
        "--dtype", cfg.get("dtype", "auto"),
    ]
    if "CUDA_VISIBLE_DEVICES" in env:
        n_devices = len(env["CUDA_VISIBLE_DEVICES"].split(","))
        vllm_cmd += ["--tensor-parallel-size", str(n_devices)]

    logger.info(f"Starting vLLM server with command: {' '.join(vllm_cmd)}")
    pid = os.getpid()
    if is_judge:
        stdout_log = f"logs/{pid}-judge.out"
        stderr_log = f"logs/{pid}-judge.err"
    else:
        stdout_log = f"logs/{pid}-vllm.out"
        stderr_log = f"logs/{pid}-vllm.err"
    logger.info(f"Logging to: {stdout_log} (STDOUT) {stderr_log} (STDERR)")
    logger.info(f"Follow logs with: tail -f logs/{pid}-vllm.*")

    with open(stdout_log, "w") as out, open(stderr_log, "w") as err:
        proc = subprocess.Popen(vllm_cmd, env=env, stdout=out, stderr=err)

    for _ in range(MAX_VLLM_START_TIMEOUT // HEALTH_CHECK_FREQ):
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM subprocess exited early with return code {proc.returncode}. Check {stdout_log} and {stderr_log} for more information.")
        try:
            resp = requests.get(f"http://localhost:{port}/health")
            if resp.status_code == 200:
                logger.info("✅ vLLM server ready!")
                return proc
            
        except requests.exceptions.ConnectionError:
            pass
        logger.info("Waiting for vLLM to start")
        time.sleep(HEALTH_CHECK_FREQ)
    else:
        logger.error(f"❌ vLLM server did not start in time (timeout: {MAX_VLLM_START_TIMEOUT}s).")
        proc.terminate()

def launch_llm(model_name: str, vllm_config: str):
    if is_local_path(model_name) or is_huggingface_model(model_name):
        chat_type = "vllm"
        proc = start_vllm_server(model_name, vllm_config) # blocks until healthy
        if proc is None:
            raise RuntimeError(f"vLLM server failed to start in time. Try pre-downloading the model weights, or file an issue.")
    elif is_openai_model(model_name):
        proc = None
        chat_type = "openai"
    else:
        raise ValueError(f"Model '{model_name}' not recognized as a local path, HuggingFace model, or OpenAI model.")
    return proc, chat_type

def launch_judge(judge_config: str):
    proc = start_vllm_server(judge_config["model"], judge_config["vllm_args"], is_judge=True) # blocks until healthy
    if proc is None:
        raise RuntimeError(f"vLLM server failed to start in time. Try pre-downloading the model weights, or file an issue.")

    return proc

GOALSPACE_START_TIMEOUT = 600 # OK, you got me. Not technically an LLM.
def block_until_healthy(proc, port: int):
    for _ in range(GOALSPACE_START_TIMEOUT // HEALTH_CHECK_FREQ):
        try:
            resp = requests.get(f"http://localhost:{port}/health")
            if resp.status_code == 200:
                logger.info("✅ Goalspace server ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(HEALTH_CHECK_FREQ)
    else:
        logger.error(f"❌ Server did not start in time (timeout: {GOALSPACE_START_TIMEOUT}s).")
        proc.terminate()
        return False

def launch_goalspace_server(uvicorn_cfg: dict):
    env = os.environ.copy()
    env["GOAL_DIMENSIONS"] = ",".join(uvicorn_cfg["goal_dimensions"])
    env["NORMALIZATION_PATH"] = uvicorn_cfg["normalization"]
    command = [
        "uvicorn",
        "steerability.goalspace_server:app",
        "--port", str(uvicorn_cfg["port"]),
        "--host", str(uvicorn_cfg["host"]),
        "--workers", str(uvicorn_cfg["workers"]),
    ]
    pid = os.getpid()
    logger.info(f"Starting goalspace server with command: {' '.join(command)}")

    stdout_log = f"logs/{pid}-uvicorn.out"
    stderr_log = f"logs/{pid}-uvicorn.err"
    logger.info(f"Logging to: {stdout_log} (STDOUT) {stderr_log} (STDERR)")
    logger.info(f"Follow logs with: tail -f logs/{pid}-uvicorn.*")
    with open(stdout_log, "w") as out, open(stderr_log, "w") as err:
        proc = subprocess.Popen(command, env=env, stdout=out, stderr=err)
    return proc


