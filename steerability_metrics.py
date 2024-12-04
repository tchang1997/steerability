import numpy as np
import pandas as pd

from typing import Optional

def extract_vectors(steer_data: pd.DataFrame, align: Optional[bool] = True):
    if align:
        base_cols = steer_data.filter(like="delta_")
    else:
        base_cols = steer_data.filter(like="source_")
    goals = [col.split("_", 1)[1] if "_" in col else col for col in base_cols]
    pattern = '|'.join([f"_{suffix}$" for suffix in goals])
    
    z0 = steer_data.filter(like="source_").filter(regex=pattern)
    z_hat = steer_data.filter(like="output_").filter(regex=pattern)
    z_star = steer_data.filter(like="target_").filter(regex=pattern)

    # align dfs
    z0 = z0[[f"source_{g}" for g in goals]]
    z_hat = z_hat[[f"output_{g}" for g in goals]]
    z_star = z_star[[f"target_{g}" for g in goals if f"target_{g}" in z_star.columns]]

    z0.columns = goals
    z_hat.columns = goals
    if align:
        z_star.columns = goals 
    else:
        z_star_goals = [col.split("_", 1)[1] if "_" in col else col for col in z_star.columns]
        z_star.columns = z_star_goals
    z_star = z_star.where(pd.notnull(z_star), z0)
    return z0, z_star, z_hat

def sensitivity(z0: pd.DataFrame, z_star: pd.DataFrame, z_hat: pd.DataFrame):
    # shape should be the same -- use z_star to align
    return np.linalg.norm(z_hat - z0, axis=1) / np.linalg.norm(z_star - z0, axis=1)

def directionality(z0: pd.DataFrame, z_star: pd.DataFrame, z_hat: pd.DataFrame):
    what_we_want = z_star - z0
    what_we_got = z_hat - z0
    dp = np.sum(what_we_want * what_we_got, axis=1)
    norm_www = np.linalg.norm(what_we_want, axis=1)
    norm_wwg = np.linalg.norm(what_we_got, axis=1)
    return dp / (norm_www * norm_wwg)
