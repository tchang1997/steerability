from argparse import ArgumentParser
from functools import reduce 
import os

import pandas as pd
from ruamel.yaml import YAML

from steerability.instruction_generator import get_instruction_generator

from typing import Optional

yaml = YAML(typ="safe")
"""
    For multi-stage setups, creating an eval probe is tricky. The training datasets are different, so one may not always
    want to evaluate on a "held-out" subset of the training data, but keep the test set fixed across runs, almost serving
    as a meta-test set. However, the source texts must be disjoint from any training dataset, and the weighted sampling strategy
    also complicates things somewhat. 
    
    This is a helper script for creating such a meta-eval probe. Given a list of training configs, and a "target probe" from which 
    the meta-eval probe should be drawn, this dataset creates a meta-eval probe by reconstructing the datasets produced, and takes the
    union of all source texts seen during training. In `rl.py`, all probes are first split into a train-eligible and test-eligible
    set of source texts, from which the final train/test probes are sampled. 

    So, we take the union of the train-eligilbe responses and exclude them from the test-eligible responses from the final stage config. 
    Texts in the meta-eval probe are filtered accordingly. We take a weighted subsample and save meta-eval probes of the requested
    size (usually N=32-64 for quick in-training checks and N=1024-2048 for formal, post-training evals).

    In theory we can save the training dataset CSVs and do a string-by-string check but this procedure makes sure that our 
    exclusion logic exactly matches the training probe generation code. 

    Note: worst case, if there's too many, we can go back to the seed texts and re-generate a few.
"""

def get_train_eligible_ids(config: str) -> set:
    with open(config, "r") as f:
        cfg = yaml.load(f)
    train_test_seed = cfg["probe_sampling_seed"]
    path = cfg["steerability_probe"]
    probe = pd.read_csv(path, index_col=0)

    # same as rl.py
    train_probe = probe.sample(frac=0.5, random_state=train_test_seed)
    return set(train_probe['original_index'])

def get_meta_test_probe(config: str) -> pd.DataFrame:
    with open(config, "r") as f:
        cfg = yaml.load(f)
    train_test_seed = cfg["probe_sampling_seed"]
    path = cfg["steerability_probe"]
    probe = pd.read_csv(path, index_col=0) 

    train_probe = probe.sample(frac=0.5, random_state=train_test_seed)
    test_probe = probe.drop(train_probe.index)
    return test_probe

def generate_instructions(
    test_data: pd.DataFrame,
    model_name: str,
    prompt_strategy: Optional[str] = "direct",
    disambig: Optional[bool] = True
):
    instruction_generator = get_instruction_generator(prompt_strategy)
    delta_goals = test_data.filter(like='delta_', axis=1) # This is heavily reliant on how the probe is implemented at an earlier stage -- target for further refactor
    target_goals = test_data.filter(like='target_', axis=1)
    source_goals = test_data.filter(like="source_", axis=1)

    for target_col in target_goals.columns:
        source_col = target_col.replace("target_", "source_")
        test_data[target_col] = target_goals[target_col].fillna(source_goals[source_col]) # patch the probe 

    test_data["instructions"] = instruction_generator.sample_prompt(delta_goals, target_goals, disambig=disambig) 
    test_data["prompt"] = test_data["instructions"] + "\n\n" + test_data["text"]
    test_data["model_name"] = model_name
    return test_data

def sample_probe(
        eligible_df: pd.DataFrame,
        size_config: str,
        random_state: Optional[int] = 430,
        model_name: Optional[str] = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    ) -> pd.DataFrame:
    n_source_texts, n_insts = map(int, size_config.split("x"))
    test_source_texts = eligible_df["text"] \
        .drop_duplicates() \
        .sample(
            n=n_source_texts,
            random_state=random_state,
            replace=False,
        )
    test_subset = eligible_df[eligible_df["text"].isin(test_source_texts)]
    test_data = test_subset.groupby("text", group_keys=False).apply(
        lambda x: x.sample(
            n=min(len(x), n_insts),
            random_state=random_state,
            replace=False,
        )
    ).reset_index(drop=True) # this'll sample the same instruction indices per source text, which shouldn't be the same as the same instructions 
    test_data = generate_instructions(test_data, model_name)
    return test_data


if __name__ == "__main__":
    psr = ArgumentParser()
    psr.add_argument("--training-configs", nargs="+", type=str, required=True)
    psr.add_argument("--final-stage-config", type=str, required=True)
    psr.add_argument("--sizes-to-sample", nargs="+", type=str, default=["8x4", "64x32"])
    psr.add_argument("--name", type=str, default="_meta_test")
    args = psr.parse_args()
 
    print("Computing train-eligible IDs...")
    train_ids = [get_train_eligible_ids(config) for config in args.training_configs]
    train_eligible_ids = reduce(set.union, train_ids)

    print("Filtering meta-test probe...")
    meta_test = get_meta_test_probe(args.final_stage_config)
    print("Meta-test probe has", len(meta_test["text"].unique()), "texts")
    eligible_meta_test = meta_test[~meta_test['original_index'].isin(train_eligible_ids)]

    print(len(eligible_meta_test["text"].unique()), "source texts remain eligible for meta-testing")
    for size_config in args.sizes_to_sample:
        df = sample_probe(eligible_meta_test, size_config)
        path = os.path.join("./data/eval", f"{args.name}_{size_config}.csv")
        if os.path.isfile(path):
            print("Path exists! Skipping...")
            continue
        df.to_csv(path)
        print("Saved meta-test probe to", path)



    
