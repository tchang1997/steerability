# Replication instructions

Thank you for your interest in replicating our work! Here, we provide instructions on how to do so.

## Generating our steerability probe

In general, you can run:
```
    python seed_data.py --config config/seed_data/seed_data_v2.yml
    python generate_steerability_probe.py --seed-data ./data/v2_seed_data.csv --config [CONFIG_FILE] --goals [GOAL_DIMENSIONS] \ # optional args follow
        --use-async --uvicorn-port 9999 --max-workers 32
```
where the config file for `generate_steerability_probe.py` used in our work can be found at `./config/probes/*.yml`. See `goals.py` for goal dimension names supported. 

Due to seeding differences across hardware, the exact probe may vary, but will be released as a Huggingface Dataset.


## Steerability reports

Please check `config/experiments` for YAML files denoting all steerability evaluations in our main experiments and Appendix. These methods can be run as follows:

```
    python create_steerability_report.py --config [YAML_file] --api-config [PATH_TO_API_CONFIG]
``` 

For non-OpenAI models, please serve locally via vLLM first at port 16384 (or a port of your choice, as per the config files), for which we defer to the (https://docs.vllm.ai/en/v0.8.4/serving/openai_compatible_server.html)[vLLM 0.8.4 documentation]. You may optionally add the `--use-async` flag for faster goalspace-mapping calculations, but you must first run:

```
    uvicorn steerability.goalspace_server:app --host 127.0.0.1 --port 16641 --workers [N_CPUS]
```

## RL fine-tuning

Our fine-tuning pipeline depends on a custom fork of `trl`. Config files describing our final RL experiments can be found at `config/rl/*.yml`. Training jobs are launched via:

```
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 python -m trl.scripts.vllm_serve --model meta-llama/Llama-3.1-8B-Instruct --port 9877 --host localhost --enable_prefix_caching True --max_model_len 4096 --dtype bfloat16 --gpu_memory_utilization 0.9 --tensor_parallel_size 2 # wait for server to start
    CUDA_VISIBLE_DEVICES=4 uvicorn goalspace_server:app --host 127.0.0.1 --port 9999 --workers 32 # wait for server to start 
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num-processes 2 --main_process_port 29503 rl.py --config [YAML_FILE]
```

and training probes will be saved at `./training_probes/` for further inspection. By default, we also log via `wandb`, which can be configured via `export WANDB_PROJECT=[your_project_name]`.

## Creating steerability plots and flow diagrams

All figures in our paper were created in `create_steerability_plots.ipynb`, including static vector flow diagrams. For dynamic flow diagrams, we recommend the `steerflow` tool (see main README)!

