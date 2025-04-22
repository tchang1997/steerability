import numpy as np
import pandas as pd

# Some helpers initially used for post-processing results during plotting.

STEERING_GOALS = ["reading_difficulty", "textual_diversity", "text_length", "formality"]

def scalar_rejection(a, b): # b onto a
    #return np.sqrt(np.linalg.norm(b, axis=1) ** 2 - (np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) + 1e-8)) ** 2)
    b_norm = np.sum(b * b, axis=1).values + 1e-8
    proj = (np.sum(a * b, axis=1) / b_norm).values.reshape(-1, 1) * b  # shape [n, d]
    rejection_vec = a - proj
    rejection = np.linalg.norm(rejection_vec, axis=1)
    return rejection

def scalar_projection(a, b):
    return np.sum(a * b, axis=1) / (np.linalg.norm(b, axis=1) + 1e-8)

def get_dist_to_goal(df, by_goal=False, steering_goals=STEERING_GOALS):
    df = df.replace(-1, np.nan) # undo the sentinel!
    source = df[[f"source_{goal}" for goal in steering_goals]]
    output = df[[f"output_{goal}" for goal in steering_goals]]
    target = df[[f"target_{goal}" for goal in steering_goals]]
    target_corrected = target.where(pd.notnull(target), source.values)
    if by_goal:
        dists = pd.DataFrame(target_corrected.values - output.values, columns=steering_goals)
    else:
        dists = np.linalg.norm(output.values - target_corrected.values, axis=1)
    return dists

def get_dist_from_source(df, by_goal=False, steering_goals=STEERING_GOALS): # \hat{z} - z_0
    source = df[[f"source_{goal}" for goal in steering_goals]]
    output = df[[f"output_{goal}" for goal in steering_goals]]           
    if by_goal:
        dists = pd.DataFrame(output.values - source.values, columns=steering_goals)
    else:
        dists = np.linalg.norm(output.values - source.values, axis=1)
    return dists


def get_dist_requested(df, by_goal=False, steering_goals=STEERING_GOALS): # z^* - z_0
    source = df[[f"source_{goal}" for goal in steering_goals]]
    target = df[[f"target_{goal}" for goal in steering_goals]]  
        
    target_corrected = target.where(pd.notnull(target), source.values)
    if by_goal:
        dists = pd.DataFrame(target_corrected.values - source.values, columns=steering_goals)
    else:
        dists = np.linalg.norm(source.values - target_corrected.values, axis=1)
    return dists

def extract_oracle_best(df, mode="best", steering_goals=STEERING_GOALS):
    target_goals = [f"target_{goal}" for goal in steering_goals] # need to also groupby target goals to avoid duplicate prompt issues!
    grouping = df.fillna(-1).groupby(["text"] + target_goals)
    if mode == "best":
        best_completions = grouping.apply(lambda group: group.iloc[get_dist_to_goal(group, steering_goals=steering_goals).argmin()]) \
            .reset_index(drop=True)
    elif mode == "worst":
        best_completions = grouping.apply(lambda group: group.iloc[get_dist_to_goal(group, steering_goals=steering_goals).argmax()]) \
            .reset_index(drop=True)
    elif mode == "median":
        def argmedian(group):
            target_dist = get_dist_to_goal(group, steering_goals=steering_goals)
            med = target_dist.median()
            med_dist = np.abs(target_dist - med)
            return med_dist.argmin()
        
        best_completions = grouping.apply(lambda group: argmedian(group)) \
            .reset_index(drop=True)
    else:
        raise ValueError()
    return best_completions


