import pandas as pd

from typing import Optional

def get_grounded_subset(df: pd.DataFrame, flip: Optional[bool] = False):
    mask = (df["rationale_approved"] & (df["answer"] == "Yes")) | (~df["rationale_approved"] & (df["answer"] == "No"))
    #print(f"{mask.sum()}/{len(df)} rewrites passed groudnedness check | {(df['answer'] == 'No').sum()} flagged | {(~df['rationale_approved']).sum()} overruled")
    #if flip:
    #    mask = ~mask
    #return df.loc[mask]
    n_flagged = (df['answer'] == 'No').sum()
    n_overruled = (~df['rationale_approved']).sum()
    return mask, n_flagged, n_overruled

def get_nonnull_targets(probe: pd.DataFrame):
    target_goals = probe.filter(like='target_', axis=1)
    source_goals = probe.filter(like="source_", axis=1)

    target_corrected_goals = target_goals.copy()
    for target_col in target_goals.columns:
        source_col = target_col.replace("target_", "source_")
        target_corrected_goals[target_col] = target_goals[target_col].fillna(source_goals[source_col])
    return target_corrected_goals