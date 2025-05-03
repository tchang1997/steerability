# Measuring Steerability In LLMs

This repository implements steerability probes and provides steerability reports for LLMs. It also provides the means to make your own steerability probes. This steerability evaluation is meant to evaluate the output-behavior of LLMs. It implements the steerability measurement framework introduced in ["Measuring Steerability in Large Language Models."](https://openreview.net/forum?id=y2J5dAqcJW) and provides basic utilities for training models to be more steerable. 

Note this is very early-stage work and will change in the future. This work was partially done as a research intern at Microsoft Research.

## Steerability Tuning

### Training

Training proceeds from a custom fork of `trl`. We leverage `accelerate` for multi-GPU fine-tuning. All our experiments use 2 GPUs for training with 2 gradient accumulation steps; YMMV if you use different settings. First, you must launch a vLLM server via `trl`:
```
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=2,3 trl vllm-serve \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 9876 --host localhost \
    --enable_prefix_caching True \
    --max_model_len 4096 \
    --dtype bfloat16 \
    --gpu_memory_utilization 0.8 \
    --tensor_parallel_size 2
```
Once that's running, this command will start a training run for a config file:
```
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --num-processes 2 \
    --main_process_port 29504 \
    rl.py --config config/rl/my_training_run.yml
```
On some machines where `/tmp` is mounted as `noexec`, you may need to set `TMPDIR` to point to a directory where you have execute permissions to play nice with some Torch Inductor + Triton stuff under the hoo

Check `config/rl` for example config files, and `utils/train_config_dataclasses.py` for settings that are outside `GRPOConfig` and `TrainingArguments` in `trl` and `transformers`, respectively. 

### Evaluation

By default, our training runs save 8 checkpoints, all of which are LoRA adapters. The save path will be specified by the `output_dir` key in the training config YAML file. You can simply do
```
bash load_lora.sh {your_model_output_dir}
```
and this will automatically skip up a vLLM instance. Under the hood, it's really running this command:
```
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
USE_FASTSAFETENSOR=true \
VLLM_USE_V1=1 \
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 16384 \
  --enable-lora \
  --dtype bfloat16 \
  --max-lora-rank 256 \
  --lora-dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.6 \
  --lora-modules $LORA_MODULES
```
*i.e.,* the same settings used during our default training runs, where `$LORA_MODULES` is generated dynamically by listing the checkpoints in a particular directory. Do note that usage of `VLLM_USE_V1=1` is obligatory here; there seems to be some device-setting bugs when `VLLM_USE_V1` is not set. If you want to change any of these defaults, you can modify `load_lora.sh` (or file a PR where we pass them in as arguments)! Then, to evaluate, you can create config files in the same format as our other experiments. Or, just do this AFTER the vLLM server is up and running:
```
python create_eval_configs.py
```
This will ping vLLM to get a list of adapters and auto-generate the config files in a subdirectory. It also generates some nice scripts `eval_train.sh` and `eval_test.sh` in the relevant subdirectories for you to run eval on the default training and test sets. 


## Steerability Reports
We design a steerability probe for comparing steerability across LLMs. In short, we define a vector-space of goals (goal-space) and model user requests and texts as vectors in goal-space. The LLM's behavior can then be thought of as a vector operation, and we can compare how the LLM's behavior vector matches with the user's request. Formally, steerability becomes a normalized dot product between "what we wanted" and "what we got." Our steerability metric is a real number, with the best value being 1. Check out [our paper](https://openreview.net/forum?id=y2J5dAqcJW) for a formal definition!

|Model|Steerability|
|-|-|
|GPT-3.5|0.298±0.461|
|GPT-4 turbo|0.344±0.470|
|Llama3-8B|0.520±0.529|
|Llama3-70B|0.521±0.511|
|Mistral-7B|0.295±0.462|
|Mixtral-8x7B|0.302±0.476|

The errors are the **standard deviation** taken over our empirical steerability probe (`steerbench_converted.csv`). 

Have a suggestion for a model for us to try? Would you like to submit your own model for evaluation? Let us know at `ctrenton` at `umich` dot `edu`! 

## Generating steerability reports

You can use the same steerability probe that we've created out of the box at `data/steerbench_converted.csv`:

```
python create_steerability_report.py --config [CONFIG] --api-config [API_KEY_FILE]
```

For examples of config files, you can check out `config/mockup.yml` or `config/gpt4_mini.yml`. Currently, we support interacting with the OpenAI API, DeepInfra API, and vLLM.

## Evaluating your own LLM
Our steerability evaluation framework supports any vLLM-supported model via their OpenAI-compatible server.   Simply write a config of the following form:
```
TODO
```

Feel free to use our pre-generated steerability probes, or generate your own below. 

## Submitting your own LLM outputs to our steerability evaluation server

TODO: We are also working on an evaluation server to allow researchers to submit LLM-generated responses to a given steerability probe for evaluation on our end. We aim to support evaluation on the following probes used in our paper:
* **SteerBench-3D-in-10D** (original)
* **SteerBench-ND-in-10D** for N = 1, 2, 4 
* **SteerBench-2D-in-3D-split_1** (used for training our models)
* **SteerBench-2D-in-3D-split_2** (used for testing our models)

We encourage submissions to focus on **SteerBench-3D-in-10D** for direct comparability with our evaluation, but release all probes publicly for transparency.  

# Conducting your own steerability analysis from scratch

## Installation

You will need to run:

1. `pip install -r requirements.txt` to install dependencies
2. `bash initial_setup.sh` to create the initial folders

## Quickstart

Run:
```
    python seed_data.py
    python create_steerability_probe.py --seed-data [OUTPUT_OF_SEED_DATA]
    python create_steerability_report.py --config [CONFIG] --api-config [API_KEY_FILE]
```

A minimal example of a config file is provided at `config/experiments/mockup.yml`. The `--api-config` should be a file that contains JSON of the form `{api_key: [API_KEY]}`.

This will generate a steerability report on our default steerability benchmark, using a direct prompting strategy and GPT-4 turbo.

## Creating a steerability probe

A steerability probe is dependent on a set of source texts and a set of goals. The general flow is as follows:

1. Create a "seed set" of source texts
2. Map texts to goal-space
3. Create a steerability probe by taking a weighted subsample of the source texts
4. Choose prompting strategy for goals 
5. Run steerability evaluation + generate reports

Generally, Steps 1-3 create a fixed dataset called a "steerability probe" that can be applied to multiple LLMs and prompting strategies. Here's how to do each step in this repo.

### Generating a seed set (Step 1)

Simply run `python seed_data.py`. This uses the configuration file in `config/seed_data/default_seed_data.yml`, which specifies a set of datasets and rules for pre-processing. We want all data to be concatenated into a one-string-per-line format, subject to some set of processing rules specified in the config.

**General format:**
```
python seed_data.py [--config CONFIG] [--reload [dataset_names ...]]
``` 

### Create a steerability probe (Steps 2 and 3)

This step will evaluate goal-space mappings for texts (or retrieve them if they are cached), curate a weighted subsample of texts of a fixed size, and save the resulting probe to disk. Settings for the probe (e.g., # of active goals, # of goals per text) can be found at `default_probe_settings.yml`. 

**General format:**
```
python generate_steerability_probe.py [--seed-data SEED_DATA_CSV] [--config CONFIG]
```

For further customization, you can specify exact goals you want so long as they are registered in `goals.py`:
```
python generate_steerability_probe.py --config config/probes/2d_steer_in_1d_uncorr.yml \
    --seed-data data/default_seed_data.csv \
    --goals sadness surprise \
    --weighting-goals sadness surprise
```

### Create a steerability report (Steps 4 and 5)

This step will, given a configuration file and a prompting strategy, then run a steerability evaluation with an LLM + generate a report (plots).

**General format:**
```
python create_steerability_report.py --config [CONFIG] --api-config [API_KEY_FILE]
[--overwrite] [--nrows N_ROWS]
```

Only the first two (experimental config and API config) are required. This will output a CSV with the raw "steerability data" (i.e., goalspace mappings of the LLM inputs and outputs).

You can test to see if the prompt strategy generations instructions that meet your expectations as follows before running an entire probe:
```
python instruction_generator.py --probe data/steer_2d.csv --prompt-strategy direct
```

### Future implementation

* A steerability dashboard where you can explore & visualize the steerability data interactively
* Interfaces with more LLMs

Please file a GitHub issue or email `ctrenton` at `umich` dot `edu` if you have any suggestions, questions, or would like to contribute!

## VLLM Setup (for pinging models locally).

This step is a little more involved. First, it's highly recommended that you create a new env for VLLM and follow their instructions. Then, you should download models via the `huggingface-cli` before actual use via `vllm serve`. The steps are:

1. `pip install huggingface[cli]`
2. Set the `HF_TOKEN` and (optionally) `HF_HOME` environment variables in your shell config. You'll need to generate `HF_TOKEN` on your HuggingFace account first.
3. Download the models you want via `huggingface-cli download [ORG_NAME/MODEL_NAME]`
4. `vllm serve [ORG_NAME/MODEL_NAME] [--your-args-here]`

You can verify that the server is running via `curl http://localhost:PORT/v1/models`. 

## Other

### LLM as judge:
Here's the command used to compare pre- vs. post-RL rewrites:
```
python utils/pairwise_goal_validation.py --probe [probe_1_path] [probe_2_path] --no-sample --source-col left_llm_response --response-col right_llm_response --source-goal-prefix left_output_ --response-goal-prefix right_output_ --name pre_vs_post_rl_best_on_test
```
Do note that this'll attach to whichever vLLM instance that is already running. 