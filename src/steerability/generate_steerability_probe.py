"""
    Ideally, this file should take the seed data and create a steerability probe (dataset of starting points + sampling weights only). 

    Inputs - Steerability probe parameters
    - # of texts to sample
    - # of goals per text

    Outputs 
    - Raw & normalized goal-space mappings
    - Normalized uniform sampling weights

"""
from argparse import ArgumentParser
import asyncio
import os

from beartype import beartype
from nltk import sent_tokenize
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from ruamel.yaml import YAML
from tqdm.auto import tqdm

from steerability.goals import Goalspace, GoalFactory, ALL_GOALS, DEFAULT_GOALS
from steerability.rewards import map_to_goalspace

from typing import Optional, Union

tqdm.pandas()
yaml = YAML(typ='safe')

@beartype
def get_uniform_weights(goalspace_df: pd.DataFrame, weighting_goals: Goalspace) -> np.ndarray:
    goal_names = ["source_" + name for name in weighting_goals.get_goal_names(snake_case=True)]
    goalspace_subset = goalspace_df[goal_names]
    random_mask = pd.DataFrame(np.random.rand(*goalspace_subset.shape), columns=goal_names)
    df = pd.concat([goalspace_subset, random_mask], keys=[1, 0], names=["real"]).reset_index(level=0)
    
    # classifier-based density-ratio estimation (Bickel, ICML '07)
    model = LogisticRegression()
    X = df.drop(columns=["real"])
    y = df["real"]
    model.fit(X, y)
    probs = model.predict_proba(X)
    overlap = roc_auc_score(y, probs[:, 1])
    print("AUC (high values indicate very non-uniform seed texts):", overlap)

    raw_weights = probs[:len(goalspace_subset), 0] / probs[:len(goalspace_subset), 1]
    return raw_weights / raw_weights.sum()

@beartype
def normalize_goals(goalspace_df: pd.DataFrame, range: Optional[Union[float, int]] = 95) -> pd.DataFrame:
    if range <= 0 or range >= 100: 
        raise ValueError("Range must be in (0, 100).")
    goals = goalspace_df.columns.copy()
    min_q = (100 - range) / 2
    max_q = 100 - min_q
    for col in goals:
        lower_bound = np.percentile(goalspace_df[col], min_q)
        upper_bound = np.percentile(goalspace_df[col], max_q)
        print(f"Goal {col} is normalized to [{lower_bound}, {upper_bound}]")
        goalspace_df[f"source_{col}"] = (goalspace_df[col] - lower_bound) / (upper_bound - lower_bound)
        goalspace_df[f"source_{col}"] = goalspace_df[f"source_{col}"].clip(0, 1)
    return goalspace_df

@beartype
def generate_target_goals(shape: tuple, max_active_goals: Optional[int] = 3) -> np.ndarray:
    valid_goal_mask = np.ones(shape, dtype=bool)
    for i in range(shape[0]): # separate loop for proposing goals -- that way we can batch pre-specified goals
        candidate_goals = np.random.choice(range(shape[1]), size=max_active_goals, replace=False) # select N goals
        valid_goal_mask[i, candidate_goals] = False 
    return valid_goal_mask

@beartype
def get_subset(
    goalspace_df: pd.DataFrame,
    n_source_texts: int, 
    n_goals_per_text: int,
    seed: Optional[int] = 42,
    weighted: Optional[bool] = True,
):
    if weighted:
        text_subset = goalspace_df.sample(n=n_source_texts, weights=goalspace_df["sampling_weights"], replace=False, random_state=seed) # yes, sampling w/o replacement will introduce a slight amount of bias, but this will reduce issues where we repeatedly sample a super "rare" text
    else:
        text_subset = goalspace_df.sample(n=n_source_texts, replace=False, random_state=seed)
    text_subset = text_subset.loc[text_subset.index.repeat(n_goals_per_text)]
    return text_subset   

@beartype
def generate_deltas(
    source_goals: np.ndarray,
    delta_min: Optional[float] = 0.1,
    delta_max: Optional[float] = 0.7,
    deadzone: Optional[float] = 0.1,
    max_active_goals: Optional[int] = 3,
    seed: Optional[int] = 42,
) -> np.ndarray:
        # In the future, the below can be refactored into arbitrary target goal generator 
    lower_ranges = np.clip(source_goals - deadzone, 0, -delta_min - deadzone)
    upper_ranges = np.clip(1 - source_goals - deadzone, 0, delta_max - deadzone)
    all_ranges = lower_ranges + upper_ranges

    # Compute the minima:
    # [-source_goals, -deadzone] U [deadzone, 1 - source_goals] (up to a global range of [delta_min, delta_max])
    # if -deadzone < -source_goals <=> source_goals < deadzone, then only the upper interval is active => set minima to `deadzone`
    # if deadzone > 1 - source_goals, then only the lower interval is active => set maxima to `-deadzone`
    # use the minima and maxima to generate a "deadzone active" mask (True if both intervals are "active")
    np.random.seed(seed)
    valid_goal_mask = generate_target_goals(source_goals.shape, max_active_goals=max_active_goals)

    all_minima = np.maximum(-source_goals, delta_min)
    all_minima[source_goals < deadzone] = deadzone
    all_maxima = np.minimum(1 - source_goals, delta_max)
    all_maxima[1 - source_goals < deadzone] = -deadzone
    deadzone_active_mask = (all_minima < deadzone) & (all_maxima > -deadzone) # whether a range crosses the deadzone
    assert np.allclose(((all_maxima - all_minima) - all_ranges)[deadzone_active_mask], 2 * deadzone) # check my math. the range should be off by the deadzone width exactly when active
    assert np.allclose(((all_maxima - all_minima) - all_ranges)[~deadzone_active_mask], 0.)

    # sample U(0, 1) and rescale based on minima and range
    deltas = np.random.rand(*source_goals.shape) * all_ranges + all_minima

    # if both intervals are active, then adjust values in the deadzone out of the deadzone
    deltas[(deltas > -deadzone) & deadzone_active_mask] += 2 * deadzone 

    # finally apply the goal mask from earlier
    final_deltas = np.ma.array(deltas, mask=valid_goal_mask)
    return final_deltas

@beartype
def create_final_probe(
        goalspace_df: pd.DataFrame,
        steering_goals: Goalspace,
        n_source_texts: int,
        n_goals_per_text: int,
        delta_min: Optional[float] = 0.1,
        delta_max: Optional[float] = 0.7,
        deadzone: Optional[float] = 0.1,
        max_active_goals: Optional[int] = 3,
        seed: Optional[int] = 42,
        weighted: Optional[bool] = True, 
    ) -> pd.DataFrame:
    text_subset = get_subset(goalspace_df, n_source_texts, n_goals_per_text, seed=seed, weighted=weighted)
    goal_names = ["source_" + name for name in steering_goals.get_goal_names(snake_case=True)]
    source_goals = text_subset[goal_names].values

    deltas = generate_deltas(
        source_goals,
        delta_min=delta_min,
        delta_max=delta_max,
        deadzone=deadzone,
        max_active_goals=max_active_goals,
        seed=seed
    )
    
    delta_cols = ["delta_" + name for name in steering_goals.get_goal_names(snake_case=True)]
    target_cols = ["target_" + name for name in steering_goals.get_goal_names(snake_case=True)]
    text_subset[delta_cols] = deltas
    text_subset[target_cols] = deltas + source_goals
    text_subset = text_subset.reset_index(names='original_index')
    return text_subset


if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--config", type=str, default="default_probe_settings.yml")
    psr.add_argument("--seed-data", required=True, type=str, help="Input seed data for which goalspace mappings should be computed.")
    psr.add_argument("--goals", type=str, nargs="+", help="Goals to evaluate.")
    psr.add_argument("--weighting-goals", type=str, nargs="+", help="Goals with respect to which sampling weights should be calculated.")
    psr.add_argument("--text-col", type=str, default="text", help="Column name storing source texts in the seed data.")
    psr.add_argument("--cache-name", type=str, default="_goalspace", help="Goalspace mapping cache file.")
    psr.add_argument("--nrows", type=int, default=None)
    psr.add_argument("--max-workers", type=int, default=1)
    psr.add_argument("--uvicorn-port", type=int, default=9999)
    psr.add_argument("--use-async", action="store_true")

    # useful for sentence based stuff
    psr.add_argument("--sent-tokenize", action="store_true")
    psr.add_argument("--normalized-data", type=str)
    args = psr.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.load(f)
    probe_settings = cfg["args"]

    dataset = pd.read_csv(args.seed_data, index_col=0)
    cache_path = os.path.join("./cache/", f"{args.cache_name}.json")
    if args.nrows is not None:
        dataset = dataset.iloc[:args.nrows]
    if args.goals is None:
        goalspace = Goalspace(DEFAULT_GOALS, cache_path=cache_path)
    else:
        goalspace = Goalspace([GoalFactory.get_default(g) for g in args.goals], cache_path=cache_path) # future: make it possible to override and pass kwargs to the downstream classes

    if args.sent_tokenize:
        df_ = dataset.sample(n=args.nrows, random_state=137).copy()
        df_["sentences"] = df_["text"].apply(sent_tokenize)
        dataset = df_[["source", "sentences"]].explode("sentences").rename(columns={"sentences": "text"}).reset_index()

    need_concat = False
    goalspace_cols = goalspace.get_goal_names(snake_case=True)
    normalized_cols = ["source_" + c for c in goalspace_cols]
    if set(goalspace_cols).issubset(set(dataset.columns)):
        print("Goal-space mappings detected; skipping recalculation.")
        goalspace_df = dataset
    else:
        need_concat = True
        if args.use_async:
            print(f"Pinging goalspace server at port {args.uvicorn_port}...")
            goalspace_dict = asyncio.run(map_to_goalspace(dataset[args.text_col].tolist(), port=args.uvicorn_port, n_workers=args.max_workers, normalize=False))
            goalspace_df = pd.DataFrame(goalspace_dict)
        else:
            goalspace_df = goalspace(dataset[args.text_col].tolist(), max_workers=args.max_workers)

    if set(normalized_cols).issubset(set(dataset.columns)):
        print("Normalized goal-space mappings detected; skipping recalculation.")
    else:
        if args.normalized_data is None:
            goalspace_df = normalize_goals(goalspace_df, range=95) # middle 95% gets linearly scaled to 0, 1
        else:
            print("Normalizing goal-space according to normalization data:", args.normalized_data)

            yaml = YAML(typ="safe")
            with open(args.normalized_data, "r") as f:
                normalization = yaml.load(f)
            for goal in goalspace_df.columns:
                goal_min, goal_max = normalization[goal]["min"], normalization[goal]["max"]
                goalspace_df[f"source_{goal}"]  = np.clip((goalspace_df[goal] - goal_min) / (goal_max - goal_min), 0., 1.)
    
    if need_concat: # not {"text", "source"}.issubset(set(goalspace_df.columns)): # if pre-existing probe has "text", "source" -- ok to skip concat
        goalspace_df = pd.concat([dataset, goalspace_df], axis=1)
    allowed_columns = ["text", "source"] + goalspace_cols + normalized_cols
    goalspace_df = goalspace_df[allowed_columns]

    if args.weighting_goals is None:
        weighting_goals = Goalspace(DEFAULT_GOALS)
    else:
        weighting_goals = Goalspace([GoalFactory.get_default(g) for g in args.weighting_goals])

    goalspace_df.loc[:, "sampling_weights"] = get_uniform_weights(goalspace_df, weighting_goals=weighting_goals) # for weights that sum to 1
    goalspace_df.loc[:, "sampling_weights_mean"] = goalspace_df["sampling_weights"] * len(goalspace_df) # for sample weight where 1 = vanilla 


    steerability_probe = create_final_probe(
        goalspace_df,
        steering_goals=weighting_goals,
        n_source_texts=probe_settings["n_source_texts"],
        n_goals_per_text=probe_settings["n_goals_per_text"],
        max_active_goals=probe_settings["max_active_goals"],
        delta_min=probe_settings["delta_min"],
        delta_max=probe_settings["delta_max"],
        deadzone=probe_settings["deadzone"],
        weighted=probe_settings.get("weighted", True),
    )
    if args.sent_tokenize:
        steerability_probe.to_csv(os.path.join("data", "_sentence_level_" + cfg["name"] + ".csv"))
    else:
        steerability_probe.to_csv(os.path.join("data", cfg["name"] + ".csv"))
        print("Saved steerability probe to", cfg["name"] + ".csv")

    if args.weighting_goals is None:
        replace_stub = "_goalspace_mapped.csv"
    elif len(args.weighting_goals) == 2:
        replace_stub = f"_goalspace_{args.weighting_goals[0]}_{args.weighting_goals[1]}.csv"
    else:
        replace_stub = f"_goalspace_{len(args.weighting_goals)}d.csv"

    if args.sent_tokenize:
        replace_stub = "_sentence_level" + replace_stub
 
    mapping_path = args.seed_data.replace(".csv", replace_stub)

    goalspace_df.to_csv(mapping_path)
    print("Saved original goalspace mappings (+/- normalization) to", mapping_path)
