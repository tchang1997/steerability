# Steerability Measurement 

This is the official repo for measuring steerability in LLMs. 

## Quickstart

You can use the goal-dimensions and steerability probes that we've already created. Download the CSV of the original steerability probe here: 

This probe has 2,048 rows consisting of 64 source texts and 32 different "goal vectors." If you're using a local model via vLLM, you'll want to start that locally, for example:

```
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 16384 --max-model-len 32000 --gpu-memory-utilization 0.9 --dtype auto 
```


You then run a command of the form 
```
python create_steerability_report.py --api-config [PATH_TO_API_KEY] --config config/experiments/BLAH.yml
```


## Steerability from scratch

To measure steerability, you need to:
* Find a list of "dimensions" you care about, that you can measure (`goals.py`)
* Find a set of "seed texts" that you'd like to steer (`seed_data.py`)
* Map those seed texts into your goal-space, and generate a steerability probe (`generate_steerability_probe.py`)

