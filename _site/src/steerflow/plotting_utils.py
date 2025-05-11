import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import json
import os

from steerability.steerability_metrics import get_dist_from_source, get_dist_requested
from steerability.utils.result_utils import STEERING_GOALS

def grab_subspace(df, *args, unspecified=True, specified_first=False, steering_goals=STEERING_GOALS):
    xy = df[[f"source_{g}" for g in args]] # can also ask for deltas
    source_df = get_dist_from_source(df, by_goal=True, steering_goals=steering_goals)
    dxdy = source_df[list(args)] # output - source
    dxdy_ideal = get_dist_requested(df, by_goal=True, steering_goals=steering_goals)[list(args)].fillna(0.)
    
    dxdy.columns = [f"d_{g}" for g in args]
    dxdy_ideal.columns = [f"d*_{g}" for g in args]
    final = pd.concat([xy, dxdy, dxdy_ideal], axis=1)
    if unspecified:
        final = final.loc[dxdy_ideal.sum(axis=1) == 0]
    elif specified_first:
        final = final.loc[(dxdy_ideal[f"d*_{args[0]}"].abs() > 0) & (dxdy_ideal[f"d*_{args[1]}"] == 0)] # TODO: make this work for >1 goal
    final.columns = ["x1", "x2", "dx1", "dx2", "dx1_ideal", "dx2_ideal"]
    return final

def export_vector_field(subspace, xcol, ycol, mode="movement", output_path="static/_field.json"):
    x1, x2 = subspace["x1"], subspace["x2"]

    if mode == "movement":
        dx1, dx2 = subspace["dx1"], subspace["dx2"]
    else:
        dx1 = subspace["dx1_ideal"] - subspace["dx1"]
        dx2 = subspace["dx2_ideal"] - subspace["dx2"]

    grid_x, grid_y = np.meshgrid(
        np.linspace(x1.min(), x1.max(), 20),
        np.linspace(x2.min(), x2.max(), 20)
    )

    grid_u = griddata((x1, x2), dx1, (grid_x, grid_y), method="linear")
    grid_v = griddata((x1, x2), dx2, (grid_x, grid_y), method="linear")

    valid = np.isfinite(grid_u) & np.isfinite(grid_v)

    field = {
        "x": grid_x.flatten()[valid.flatten()].tolist(),
        "y": grid_y.flatten()[valid.flatten()].tolist(),
        "u": grid_u.flatten()[valid.flatten()].tolist(),
        "v": grid_v.flatten()[valid.flatten()].tolist(),
        "source_x": subspace["x1"].tolist(),
        "source_y": subspace["x2"].tolist(),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(field, f)
