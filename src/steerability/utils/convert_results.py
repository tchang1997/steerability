from argparse import ArgumentParser
from pathlib import Path
import subprocess

import pandas as pd

from steerability.steerability_metrics import main_steerability_evaluation
from steerability.utils.config_utils import has_negprompt, load_yaml
from steerability.utils.io_utils import safe_json_dump
from steerability.utils.result_utils import add_run_info_to_stats, print_steerability_summary, STEERING_GOALS

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# convert old-dev-style results format (judged CSV + YAML config) to new public-facing release format, and copy results
def get_repo_root() -> Path: # :/ :/ :/ :/ whyyyyyyy is there not an easier way??
    return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode("utf-8").strip())

def load_and_convert_config(config_path: str, ):
    cfg = load_yaml(config_path)
    return {
        "model_id": cfg["llm_settings"]["llm_name"],
        "prompt_strategy": cfg["prompt_strategy"], 
        "negprompt": has_negprompt(cfg),
        "probe": cfg["probe"],
    }

if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--config", required=True, type=str, help="YAML file used to generated judged results.")
    psr.add_argument("--results", required=True, type=str, help="Post-judged results CSV.")
    psr.add_argument("--judge-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    psr.add_argument("--goal-dims", type=str, nargs="+", default=STEERING_GOALS)
    psr.add_argument("--save", required=True, type=str)
    args = psr.parse_args()

    # Load and convert config
    result_dir = get_repo_root() / "results"
    judged_path = result_dir / "judged" / (args.save + ".csv")
    final_metrics_path = result_dir / "steerability_metrics" / (args.save + ".json")
    converted_cfg = load_and_convert_config(args.config)

    judged_probe = pd.read_csv(args.results, index_col=0)
    if not judged_path.is_file():
        logging.info(f"Saving CSV to {judged_path}")
        judged_probe.to_csv(judged_path)

    steer_stats = main_steerability_evaluation(judged_probe, True, args.goal_dims)
    steer_stats = add_run_info_to_stats(converted_cfg, {"model": args.judge_model}, steer_stats)
    steer_stats["run"]["legacy_converted"] = True # converted so we can keep track
    print_steerability_summary(steer_stats)

    if not final_metrics_path.is_file():
        logging.info(f"Saving JSON to {final_metrics_path}")
        safe_json_dump(steer_stats, final_metrics_path, indent=4)
    else:
        logging.info(f"Skipping JSON save: results already found at {final_metrics_path}.")

