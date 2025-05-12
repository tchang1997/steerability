import functools

import numpy as np
import pandas as pd

from steerability.utils.probe_utils import get_grounded_subset

DIRECT_PROMPT_CUT_POINTS = np.array([-1., -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.])
DEFAULT_STEERING_GOALS = ["reading_difficulty", "textual_diversity", "text_length", "formality"]

def bin_delta(delta, cutpoints):
    return np.digitize(delta, cutpoints[1:-1], right=False)

def get_dynamic_cutpoints(start_vector, cut_points=DIRECT_PROMPT_CUT_POINTS):
    # (N, d, 1) * (1, 1, C) => (N, d, C)
    return np.clip(start_vector[..., None] + cut_points[None, None, :], 0, 1)

def scalar_rejection(a, b): # b onto a
    b_norm = np.sum(b * b, axis=1).values + 1e-8
    proj = (np.sum(a * b, axis=1) / b_norm).values.reshape(-1, 1) * b  # shape [n, d]
    rejection_vec = a - proj
    rejection = np.linalg.norm(rejection_vec, axis=1)
    return rejection

def scalar_projection(a, b):
    return np.sum(a * b, axis=1) / (np.linalg.norm(b, axis=1) + 1e-8)


def get_dist_to_goal(df, by_goal=False, steering_goals=DEFAULT_STEERING_GOALS, cut_points=DIRECT_PROMPT_CUT_POINTS):
    source = df[[f"source_{goal}" for goal in steering_goals]]
    output = df[[f"output_{goal}" for goal in steering_goals]]
    target = df[[f"target_{goal}" for goal in steering_goals]]
    target_corrected = target.where(pd.notnull(target), source.values)
    if cut_points is None:
        if by_goal:
            dists = pd.DataFrame(target_corrected.values - output.values, columns=steering_goals)
        else:
            dists = np.linalg.norm(output.values - target_corrected.values, axis=1)
    else:
        dynamic_cutpoints = get_dynamic_cutpoints(source.values)
        target_bins = np.sum(target_corrected.values[..., None] >= dynamic_cutpoints, axis=2) - 1
        obs_bins = np.sum(output.values[..., None] >= dynamic_cutpoints, axis=2) - 1
        if by_goal:
            dists = pd.DataFrame((obs_bins - target_bins) / (len(cut_points) - 1), columns=steering_goals)
        else:
            dists = np.linalg.norm((obs_bins - target_bins) / (len(cut_points) - 1), axis=1)
    return dists                 

@functools.wraps(get_dist_to_goal)
def get_steering_error(*args, **kwargs): # alias for get_dist_to_goal
    return get_dist_to_goal(*args, **kwargs)

def get_dist_from_source(df, by_goal=False, steering_goals=DEFAULT_STEERING_GOALS, cut_points=DIRECT_PROMPT_CUT_POINTS): # \hat{z} - z_0
    source = df[[f"source_{goal}" for goal in steering_goals]]
    output = df[[f"output_{goal}" for goal in steering_goals]]     
    if cut_points is None:      
        if by_goal:
            dists = pd.DataFrame(output.values - source.values, columns=steering_goals)
        else:
            dists = np.linalg.norm(output.values - source.values, axis=1)
    else:
        # number of bins actually moved
        assert len(cut_points) % 2 == 0 # if there is a "null" bin, bin # is odd, so cutpoint count is even
        bins_per_side = (len(cut_points) - 2) / 2
        dynamic_cutpoints = get_dynamic_cutpoints(source.values)
        output_bins = np.sum(output.values[..., None] >= dynamic_cutpoints, axis=2) - 1 - bins_per_side # in the 7-bin direct prompt, bin 3 is always the original source text
        if by_goal:
            dists = pd.DataFrame(output_bins / (len(cut_points) - 1), columns=steering_goals)
        else:
            dists = np.linalg.norm(output_bins / (len(cut_points) - 1), axis=1)
    return dists


def get_dist_requested(df, by_goal=False, steering_goals=DEFAULT_STEERING_GOALS, cut_points=DIRECT_PROMPT_CUT_POINTS): # z^* - z_0
    source = df[[f"source_{goal}" for goal in steering_goals]]
    target = df[[f"target_{goal}" for goal in steering_goals]]  
        
    target_corrected = target.where(pd.notnull(target), source.values)
    if cut_points is None:
        if by_goal:
            dists = pd.DataFrame(target_corrected.values - source.values, columns=steering_goals)
        else:
            dists = np.linalg.norm(source.values - target_corrected.values, axis=1)
    else:
        # number of bins of movement requested 
        assert len(cut_points) % 2 == 0
        bins_per_side = (len(cut_points) - 2) / 2
        delta = df[[f"delta_{goal}" for goal in steering_goals]]
        delta_corrected = delta.fillna(0.).values
        delta_bins = np.sum(delta_corrected[..., None] >= cut_points, axis=2) - 1 - bins_per_side
        if by_goal:
            dists = pd.DataFrame(delta_bins / (len(cut_points) - 1), columns=steering_goals)
        else:
            dists = np.linalg.norm(delta_bins / (len(cut_points) - 1), axis=1)
    return dists

def get_orthogonality(df, steering_goals=DEFAULT_STEERING_GOALS, normalize=True, cut_points=DIRECT_PROMPT_CUT_POINTS):
    dist_to_goal = get_dist_to_goal(df, by_goal=True, steering_goals=steering_goals, cut_points=cut_points)
    dist_from_goal_actual = get_dist_from_source(df, by_goal=True, steering_goals=steering_goals, cut_points=cut_points)
    dist_requested = get_dist_requested(df, by_goal=True, steering_goals=steering_goals)

    dist_from_goal_norm = np.linalg.norm(dist_from_goal_actual, axis=1) 
    dist_requested_norm = np.linalg.norm(dist_requested, axis=1)

    denom = dist_from_goal_norm + 1e-4 * (dist_requested_norm + 1e-8)
    rej = scalar_rejection(dist_to_goal, dist_requested)
    if normalize:
        ortho = np.clip(rej / denom, 0, 1) 
    else:
        ortho = rej
    return ortho


def get_miscalibration(df, steering_goals=DEFAULT_STEERING_GOALS, normalize=True, cut_points=DIRECT_PROMPT_CUT_POINTS):
    dist_to_goal = get_dist_to_goal(df, by_goal=True, steering_goals=steering_goals, cut_points=cut_points)
    dist_from_goal_ideal = get_dist_requested(df, by_goal=True, steering_goals=steering_goals, cut_points=cut_points)
    denom = np.linalg.norm(dist_from_goal_ideal, axis=1) + 1e-8
    abs_proj_dist = np.abs(scalar_projection(dist_to_goal, dist_from_goal_ideal))
    if normalize:
        miscal = abs_proj_dist / denom
    else:
        miscal = abs_proj_dist 
    return miscal

def get_metric_dict(arr):
    return {
        "mean": np.mean(arr),
        "median": np.median(arr),
        "min": np.min(arr),
        "max": np.max(arr),
        "std": np.std(arr),
        "iqr": np.percentile(arr, 75) - np.percentile(arr, 25), 
        "raw": arr.tolist(),
    }

def main_steerability_evaluation(probe: pd.DataFrame, judged: bool, steering_goals: list[str]):
    data_stats = {
        "n_total": len(probe),
    }
    if judged:
        grounded_mask, n_flagged, n_overruled = get_grounded_subset(probe)
        probe = probe.loc[grounded_mask]
        data_stats.update({
            "n_grounded": int(grounded_mask.sum()),
            "n_flagged": int(n_flagged),
            "n_overruled": int(n_overruled),
        })
    steering_error = get_steering_error(probe, steering_goals=steering_goals)
    miscalibration = get_miscalibration(probe, steering_goals=steering_goals, normalize=True) 
    orthogonality = get_orthogonality(probe, steering_goals=steering_goals, normalize=True)

    steer_stats = {
        "steering_error": get_metric_dict(steering_error),
        "miscalibration": get_metric_dict(miscalibration),
        "orthogonality": get_metric_dict(orthogonality),
    }
    return {
        "steerability": steer_stats,
        "data": data_stats,
    }