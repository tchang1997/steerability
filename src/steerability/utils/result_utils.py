import numpy as np
import pandas as pd

from steerability.steerability_metrics import get_dist_to_goal

# Some helpers initially used for post-processing results during plotting.

STEERING_GOALS = ["reading_difficulty", "textual_diversity", "text_length", "formality"]

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


