from argparse import ArgumentParser
import pandas as pd

if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--probe", type=str, required=True)
    psr.add_argument("--seed", type=str, default="./data/v2_seed_data_goalspace_mapped.csv")
    psr.add_argument("--null-goals", type=str, nargs="+", required=True)
    args = psr.parse_args()

    seed = pd.read_csv(args.seed, index_col=0)
    probe = pd.read_csv(args.probe, index_col=0)
    seed_to_add = seed.iloc[probe["original_index"]]
    if not (seed.iloc[probe["original_index"]].text.values == probe.text.values).all():
        raise ValueError("Texts in seed set do not match probe based on original index!")
    
    goals_added = []
    for goal in args.null_goals:
        if goal in seed_to_add.columns:
            probe[goal] = seed_to_add[goal].values
            probe[f"source_{goal}"] = seed_to_add[f"source_{goal}"].values
            probe[f"delta_{goal}"] = 0.
            probe[f"target_{goal}"] = seed_to_add[f"source_{goal}"].values
            goals_added.append(goal)
        else:
            print("Goal", goal, "not found in seed data.")
    goal_str = "_".join(goals_added)
    new_probe_path = args.probe.replace(".csv", f"_null_{goal_str}.csv")
    probe.to_csv(new_probe_path)
    print("Saved probe with added goals", goals_added, "to", new_probe_path)