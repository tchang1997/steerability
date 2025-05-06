import os
from pathlib import Path
import requests
import subprocess
import time

import openai
from openai import OpenAI
from ruamel.yaml import YAML
yaml = YAML(typ="safe")

import logging
logger = logging.getLogger(__name__)

MAX_VLLM_START_TIMEOUT = 7200
HEALTH_CHECK_FREQ = 10

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

def start_vllm_server(model_name: str, cfg: str):
    env = os.environ.copy()
    env["NCCL_P2P_DISABLE"] = "1"
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    if is_local_path(model_name):
        env["USE_FASTSAFETENSOR"] = "true"

    port = str(cfg["port"])
    vllm_cmd = [
        "vllm", "serve", model_name,
        "--port", port,
        "--max-model-len", str(cfg["max_model_len"]),
        "--gpu-memory-utilization", str(cfg["gpu_memory_utilization"]),
        "--dtype", cfg.get("dtype", "auto"),
    ]

    logger.info(f"Starting vLLM server with command: {' '.join(vllm_cmd)}")
    pid = os.getpid()
    with open(f"logs/{pid}-vllm.out", "w") as out, open(f"logs/{pid}-vllm.err", "w") as err:
        proc = subprocess.Popen(vllm_cmd, env=env, stdout=out, stderr=err)

    for _ in range(MAX_VLLM_START_TIMEOUT // HEALTH_CHECK_FREQ):
        try:
            resp = requests.get(f"http://localhost:{port}/health")
            if resp.status_code == 200:
                logger.info("✅ vLLM server ready!")
                return proc
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(HEALTH_CHECK_FREQ)
    else:
        logger.error(f"❌ vLLM server did not start in time (timeout: {MAX_VLLM_START_TIMEOUT}s).")
        proc.terminate()



