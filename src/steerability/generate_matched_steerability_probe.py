# Create a probe w/ the exact same source texts. 
from argparse import ArgumentParser

import pandas as pd
from ruamel.yaml import YAML

from generate_steerability_probe import generate_deltas

yaml = YAML(typ='safe')

if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--probe", type=str, required=True)
    psr.add_argument("--new-probe-config", type=str, required=True)
    psr.add_argument("--seed", default=42, type=int)
    args = psr.parse_args()

    probe = pd.read_csv(args.probe, index_col=0)
    with open(args.new_probe_config, "r") as f:
        cfg = yaml.load(f)
    probe_settings = cfg["args"]

    # drop delta_ and target_ cols
    raw_probe = probe.drop(probe.filter(regex="^(delta_|target_)").columns, axis=1)
    goal_names = [c.replace("source_", "") for c in raw_probe.columns if c.startswith("source_")]
    source_goals = raw_probe[["source_" + g for g in goal_names]].values

    deltas = generate_deltas(
        source_goals,
        max_active_goals=probe_settings["max_active_goals"],
        delta_min=probe_settings["delta_min"],
        delta_max=probe_settings["delta_max"],
        deadzone=probe_settings["deadzone"],
        seed=args.seed
    )

    delta_cols = ["delta_" + name for name in goal_names]
    target_cols = ["target_" + name for name in goal_names]
    raw_probe[delta_cols] = deltas
    raw_probe[target_cols] = deltas + source_goals
    name = cfg["name"]
    raw_probe.to_csv(f"./data/{name}.csv")

