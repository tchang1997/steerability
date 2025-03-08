# Steerability Tuning

This repo uses a custom branch of `trl`, `vllm`, and a `uvicorn` server acting as a reward model (i.e., for goal-space mappings). To run, you need a machine with >4 GPUs. We use A6000 GPUS (48 GB).

## Setup

Install `vllm` with `uv` following the vLLM documentation. Then:
```
pip install deepspeed accelerate 
```

Make sure you run `accelerate config` **on reboot**.

**Other gotchas:**
* Gradient checkpointing must be on with `reentrant: False`.
* You can get away with not offloading the optimizer to CPU; this is extremely risky though and might OOM deep into a run.

## Running

First, we need to spin up the goal-space mapping server. We use an external server to act as a goal-space mapper because it's meant to be a black box: goal-space is, in practice, measured by some amalgamation of pre-trained models, rules-based functions, and more, and aggregated in arbitrary ways that might not be easy to implement within a model training loop (let alone a distributed one). You can run:
```
CUDA_VISIBLE_DEVICES=[optional but recommended] uvicorn goalspace_server:app --host 127.0.0.1 --port [PORT] --workers [NUM_WORKERS]
```

If the goal-space mapper includes GPU-bound models (which it does by default), pay attention to `--workers`. In practice, we find that 2-4 workers is generally enough.

The server has endpoint `goalspace/` and simply takes in a list of strings and outputs a dictionary of goals to a list of corresponding floats. We use `aiohttp` and `asyncio` in the training loop (reward function calculation only) to speed up the process, running multiple copies of the server.

Then, start training via:
```
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=[at least 3 GPUs] accelerate launch --num-processes 2 rl.py --config [CONFIG]
```
By default, we use `deepspeed` to do ZeRO stage 2 (gradients are partitioned, but not model weights), as well as gradient checkpointing. This is sufficient to *barely* not OOM on a 48GB GPU.

This will train on two GPUs, leaving a third for vLLM inference (for generating completions to compute advantages). We use prompt-caching while generating the groups since the gradient doesn't need to propagate back; see [LINK].