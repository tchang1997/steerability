# Steerability Reporting

This repository implements steerability probes and provides steerability reports for LLMs. It also provides the means to make your own steerability probes.

This steerability evaluation is meant to evaluate the output-behavior of LLMs. It is the official implementation of the steerability measurement framework introduced in ["Measuring Steerability in Large Language Models."](TODO)

## Steerability Reports

|Model|Prompt Strategy|Steerability Score|Sensitivity|Directionality|


The error bar is the **standard deviation** taken over the empirical steerability probe (`steerbench.csv`). 

Have a suggestion for a model for us to try? Would you like to submit your own model for evaluation? Let us know at `ctrenton` at `umich` dot `edu`! 

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
python create_steerability_probe.py [--seed-data SEED_DATA_CSV] [--config CONFIG]
```

### Create a steerability report (Steps 4 and 5)

This step will, given a configuration file and a prompting strategy, then run a steerability evaluation with an LLM + generate a report (plots).

**General format:**
```
python create_steerability_report.py --config [CONFIG] --api-config [API_KEY_FILE]
[--overwrite] [--nrows N_ROWS]
```

Only the first two (experimental config and API config) are required. This will output a CSV with the raw "steerability data" (i.e., goalspace mappings of the LLM inputs and outputs), and generate plots.

By default, the above command will both conduct a steerability evaluation and generate plots, 

### Future implementation

* A steerability dashboard where you can explore & visualize the steerability data interactively

Please file a GitHub issue or email `ctrenton` at `umich` dot `edu` if you have any suggestions, questions, or would like to contribute! 
