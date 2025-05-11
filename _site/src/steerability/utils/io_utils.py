import json
from pathlib import Path

import pandas as pd  

def safe_to_csv(df: pd.DataFrame, path: str | Path, verbose: bool = True, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, **kwargs)
    if verbose:
        print(f"Saved CSV to {path}")

def safe_json_dump(obj, path: str | Path, verbose: bool = True, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, **kwargs)
    if verbose:
        print(f"Saved JSON to {path}")