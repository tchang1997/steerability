# Steerability Measurement 

This is the official repo for measuring steerability in LLMs. 

## Quickstart

First, download the CSV of the original steerability probe here: [TODO]

Simply run the following:
```
python steer_eval.py --config [YOUR_CONFIG]
```

Supported inference providers:
* OpenAI API
* vLLM self-hosted models

Supported model families:
* OpenAI API-accessible models (GPT series, o1/o3)
* Llama3
* Deepseek-R1
* Qwen3

This repo is very early stage and likely will change without notice. Issues and contributions welcome! 

## Steerability from scratch

To measure steerability, you need to:
* Find a list of "dimensions" you care about, that you can measure (`goals.py`)
* Find a set of "seed texts" that you'd like to steer (`seed_data.py`)
* Map those seed texts into your goal-space, and generate a steerability probe (`generate_steerability_probe.py`).
* From there, you can follow the quickstart. 

The guide for building a steerability probe from scratch is under construction. 