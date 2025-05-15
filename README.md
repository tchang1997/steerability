# Measuring the steerability of large language models

Welcome to the official open-source evaluation framework for measuring steerability in LLMs. 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

[Website](https://steerability.org/) | [Demo (research preview)](https://steerability.onrender.com/) | [Dataset](https://huggingface.co/datasets/tchang97/steerbench) 

![Steerflow Demo](src/steerflow/preview.gif)

*demo via [Steerflow](https://steerability.onrender.com/)*



*If you are interested in replicating the empirical analyses of our paper more closely, please consult `./src/steerability/REPLICATION.md`!*

## Installation

We recommend `uv` as the package manager. Start by running
```
uv venv /path/to/your/env/ --python 3.12.8 --seed # recommended version
source /path/to/your/env/bin/activate
uv pip install -e .
bash initial_setup.sh # makes result directories, downloads auxiliary data
```

## Quickstart

First, in a directory of your choice, create a plain text file containing your OpenAI or vLLM API key (you can set the latter yourself):
```
{"api-key": "sk-N0taR3a1AP1k3y"}
```

Then, simply run the following:
```
CUDA_VISIBLE_DEVICES=... python steer_eval.py --config [YOUR_CONFIG] --api-config [API_CONFIG]
```
where `--api-config` points to the API key file described above. 

In the provided example at `config/qwen3_example.yml`, we run the steerability probe end-to-end on Qwen3-0.6B for demonstration. This takes ~30 minutes total. **By default, the script requires manual review of rewritten texts flagged by the LLM-as-judge.** When you're finished, you'll see a print-out of key metrics. You can also play with the flow diagrams here:
```
steerflow launch --port 12347
```

We host a lightweight demo of `steerflow` [here](https://steerability.onrender.com/) as well. You can visit [our website](https://steerability.org/) for a little more info about getting started with this repo.  

**Supported inference providers:**
* OpenAI API
* vLLM self-hosted models

In theory, any vLLM-able model should work with this repo. Here's what we've tried: 
* OpenAI API-accessible models (GPT series, o1/o3)
* Llama3
* Deepseek-R1 (distilled)
* Qwen3
* Gemma3
* Phi4
* Mistral3

This repo is very early stage and likely will change without notice. Issues and contributions welcome! 

## Common issues

**Q:** The script just outputs `Waiting for vLLM to start`. Is that normal?\
**A:** If you're downloading or using a large model, it can take a while for the download/weight loading to complete. Check the log files (`tail -f logs/[PID]-vllm.*`) for the full logging output, and if it's on a download/weight-loading step, that's the issue.

To confirm that the model is simply loading, you should see something like 
```
Loading safetensors checkpoint shards:   0% Completed | 0/## [00:00<?, ?it/s]
```

**Q:** I'm sure I've downloaded the model and it still won't load after >30 min. -- how can I fix this?\
**A:** First, check the logs. You might see:
* Something about a bad request due to context length -> decrease `max_model_len` in `config/vllm_defaults/openai_server.yml` and try again
* Out of memory issues -> Try changing `gpu_memory_utilization` in `config/vllm_defaults/openai_server.yml` and try again, or set `CUDA_VISIBLE_DEVICES=...` to use multiple GPUs.
* Some torch inductor bug about not being able to import objects -> Try setting `TMPDIR=...` to a directory where you have `rwx` permissions.

Note that we've most extensively tested this script for single-GPU models — multi-GPU models can be a little finnicky, but you can try:
* Setting `NCCL_P2P_DISABLE=1` explicitly. 
* Try launching the server manually via `vllm serve` directly in your terminal.
* `vllm serve` runs fine -> our problem — please [file an issue](https://github.com/tchang1997/steerability/issues) with the name of the model you're trying to run and the command you used
* `vllm serve` also fails -> check your settings, or potentially a vLLM bug.


## Steerability from scratch

*This guide for building a steerability probe from scratch is under construction.*

All paths are relative to the `steerability` module subfolder (*i.e.*, `cd src/steerability` from here). 

To measure steerability, you need to:
* Find a list of "dimensions" you care about, that you can measure (`goals.py`)
* Find a set of "seed texts" that you'd like to steer (`seed_data.py`)
* Map those seed texts into your goal-space, and generate a steerability probe (`generate_steerability_probe.py`).
* From there, you can follow the quickstart. 

If you're just interested in using the goal dimensions we already support, here's how we generated our probe. First, we pre-processed seed data in line with some filtering rules:
```
    python seed_data.py --config config/seed_data/seed_data_v2.yml
```
In general, `config/seed_data/*.yml` files should name the HuggingFace datasets of interest, and the columns containing source texts. Examples of pre-processing rules can be found in the example YAML file; *e.g.*, de-duplication, paragraph-level chunking, min/max length filtering.

Second, we mapped all of the seed texts to goal-space and computed uniform sampling weights:
```
    python generate_steerability_probe.py --seed-data ./data/v2_seed_data.csv --config [CONFIG_FILE] --goals [GOAL_DIMENSIONS] \ # optional args follow
        --use-async --uvicorn-port 9999 --max-workers 32
```
where the config file for `./src/steerability/generate_steerability_probe.py` used in our work can be found at `./src/steerability/config/probes/*.yml`. See `./src/steerability/goals.py` for goal dimension names supported. `--use-async` can be passed for a massive speedup, but you need to manually spin up a goalspace-mapping server first:
```
    uvicorn steerability.goalspace_server:app --host 127.0.0.1 --port [PORT] --workers [NUM_CPUS]
```
This'll get you a CSV that can be directly used in `steer_eval.py` (replace the `probe` key with your CSV in the example config files). 

## Feature Support Roadmap

| Feature                                             | Support Level |
|-----------------------------------------------------|---------------|
| End-to-end evaluation                               | ✅ Fully supported + documented |
| Baseline set of prompt strategies                   | ✅ Fully supported + documented |
| Skip LLM as judge (set-and-forget mode)             | ✅ Supported  |
| Generating steerability probes from custom datasets | ⚠️ Supported but undocumented  |
| Custom goals                                         | ⚠️ Supported but undocumented |
| Custom prompt strategies                            | ⚠️ Supported but undocumented  |
| RL-based fine-tuning                                | ⚠️ Supported but undocumented  |

While there are scripts supporting most of the above features, they have not been well-tested, and dependencies may differ from those in `requirements.txt`. Please an issue or reach out for support if you're interested in trying these features.

## Citation

If you find our work or this repo useful, please cite our work:
```
@misc{chang2025steerability,
    [FORTHCOMING]
}
```

## Contact

For bug reports or feature requests, please file an issue first. 

**Email:** `ctrenton` at `umich` dot `edu`
