from argparse import ArgumentParser
from pathlib import Path
import random

import numpy as np
import pandas as pd
from ruamel.yaml import YAML

from instruction_generator import get_instruction_generator
from llm_interactor import LLMInteractor
from plotting import create_steerability_report

yaml = YAML(typ='safe')

RESULTS_DIR = Path("results/")


def get_args():
    psr = ArgumentParser()
    psr.add_argument("--config", required=True, type=str, help="YAML configuration file.")
    psr.add_argument("--api-config", required=True, type=str, help="File storing API key.")
    psr.add_argument("--inst-database", type=str, help="Auxiliary data file for advanced instructions (e.g., CoT grounded in specific examples)")
    psr.add_argument("--seed-data", type=str, help="Seed data used for steerability probe for normalization.", default="./data/default_seed_data_goalspace_mapped.csv")
    psr.add_argument("--nrows", type=int, help="Number of rows of the steerability probe to read. Useful for debugging.")
    psr.add_argument("--overwrite", action="store_true")
    return psr.parse_args()


if __name__ == '__main__':
    args = get_args()
    with open(args.config) as f:
        cfg = yaml.load(f)

    result_path = RESULTS_DIR / (cfg["experiment_name"] + ".csv")
    if result_path.is_file() and not args.overwrite:
        raise RuntimeError(f"The `--overwrite` flag is not set, and results already exist at {str(result_path)}.")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    probe_path = cfg["probe"]
    print("Loading steerability probe from", probe_path)
    probe = pd.read_csv(probe_path, index_col=0, nrows=args.nrows)
    seed_data = pd.read_csv(args.seed_data, index_col=0)

    prompt_strategy = cfg["prompt_strategy"]
    print(f"Using prompt strategy `{prompt_strategy}`")
    instruction_generator = get_instruction_generator(
        prompt_strategy,
        database=args.inst_database,
        prompter_kwargs=cfg.get("prompter_kwargs", {})
    )

    llm_cfg = cfg["llm_settings"]
    llm = LLMInteractor(
        instruction_generator,
        llm_cfg["llm_name"],
        llm_cfg["chat_type"],
        llm_cfg["cache_file"],
        args.api_config,
        **llm_cfg.get("other_kwargs", {}),
    )
    delta_goals = probe.filter(like='delta_', axis=1) # This is heavily reliant on how the probe is implemented at an earlier stage -- target for further refactor
    target_goals = probe.filter(like='target_', axis=1)
    source_goals = probe.filter(like="source_", axis=1)

    target_corrected_goals = target_goals.copy()
    for target_col in target_goals.columns:
        source_col = target_col.replace("target_", "source_")
        target_corrected_goals[target_col] = target_goals[target_col].fillna(source_goals[source_col])

    print("Writing prompts...")
    prompts = llm.write_prompts(delta_goals, target_corrected_goals, **cfg.get("inst_addons", {})) # currently, none of the implemented prompting strategies use "target_goals" but maybe just generate this w/o the weird pd.where in future probes

    print("Evaluating steerability...")
    print("Pinging endpoint:", llm.url_endpoint)
    outputs = llm.generate_steerability_data(probe, prompts, seed_data) # seed_data for normalization only -- refactor someday?

    outputs.to_csv(result_path)
    print("Steerability data saved to", result_path)

    # and now we make the plots
    # create_steerability_report(
    #   outputs,
    #    ["histogram", "goal_by_goal", "side_effect"],
    #) 
    # TODO: new eval goes here + new figure outputs
