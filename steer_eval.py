from argparse import ArgumentParser
from pathlib import Path
import pandas as pd

from steerability.llm_launcher import launch_llm 
from steerability.steerability_metrics import main_steerability_evaluation
from steerability.steerability_runners import launch_steerability_eval, run_interactive_llm_as_judge
from steerability.utils.config_utils import load_yaml
from steerability.utils.io_utils import safe_json_dump, safe_to_csv
from steerability.utils.result_utils import add_run_info_to_stats, print_steerability_summary

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("sammo").setLevel(logging.WARNING)

RESULT_DIR = Path("./results")    

def get_args():
    psr = ArgumentParser()
    psr.add_argument("--config", type=str, required=True, help="Config file for steerability evaluation.")
    psr.add_argument("--api-config", type=str, help="JSON containing API key; e.g., {'api_key': 'sk-...'}")
    psr.add_argument("--vllm-config", type=str, default="./config/vllm_defaults/openai_server.yml", help="vLLM server config file.")
    psr.add_argument("--uvicorn-config", type=str, default="./config/uvicorn_defaults/goalspace_server.yml", help="Goalspace-mapping server config.")
    psr.add_argument("--skip-judge", action="store_true", help="Whether to skip the interactive LLM as judge phase.")
    psr.add_argument("--judge-config", type=str, default="./config/vllm_defaults/judge_model.yml", help="vLLM judge model config.")
    
    psr.add_argument("--redo-inference", action="store_true", help="Whether to redo inference. Will also redo LLM-as-judge.")
    psr.add_argument("--redo-judge", action="store_true", help="Whether to redo LLM-as-judge.")
    psr.add_argument("--skip-interactive", action="store_true", help="Skip interactive review of LLM-as-judge reasoning; i.e., defer completely to the LLM's choice. We recommend leaving this flag unset.")
    args = psr.parse_args()
    return args

def run_eval_phase(result_path, probe, api_config, cfg, vllm_cfg, uvicorn_cfg, redo):
    if Path(result_path).is_file() and not redo:
        logger.info("Steerability evaluation results already found at %s", result_path)
        raw_probe = pd.read_csv(result_path, index_col=0)
    else:
        exception = None
        proc = None
        try:
            proc, chat_type = launch_llm(cfg["model_id"], vllm_cfg)
            raw_probe = launch_steerability_eval(probe, chat_type, api_config, cfg, vllm_cfg, uvicorn_cfg) # incl. reprompt if 5XX
            safe_to_csv(raw_probe, result_path)
        except Exception as e:
            exception = e
        finally:
            if proc is not None:
                if proc.poll() is None:
                    proc.terminate()
                    logger.info("Sent SIGTERM to vLLM instance.")
            if exception is not None:
                raise exception
    return raw_probe

def run_judge_phase(judged_path, judge_cfg, probe, api_config, redo, skip_interactive=False):
    if Path(judged_path).is_file() and not redo:
        logger.info("Judge results already found at %s", judged_path)
        reviewed = pd.read_csv(judged_path, index_col=0)
    else:
        reviewed = run_interactive_llm_as_judge(judge_cfg, probe, api_config, skip_interactive=skip_interactive)
        safe_to_csv(reviewed, judged_path)
    return reviewed

if __name__ == '__main__': 
    args = get_args()
    cfg = load_yaml(args.config)
    vllm_cfg = load_yaml(args.vllm_config)
    uvicorn_cfg = load_yaml(args.uvicorn_config)
    judge_cfg = None 

    name = cfg["save_as"]
    result_path = RESULT_DIR / "raw" / (name + ".csv")
    judged_path = RESULT_DIR / "judged" / (name + ".csv")
    final_metrics_path = RESULT_DIR / "steerability_metrics" / (name + ".json")

    probe = None
    probe_path = cfg["probe"]
    if not Path(probe_path).is_file():
        raise FileNotFoundError(f"Steerability probe not found at {probe_path}.")
    else:
        probe = pd.read_csv(probe_path, index_col=0, low_memory=False)

    raw_probe = run_eval_phase(result_path, probe, args.api_config, cfg, vllm_cfg, uvicorn_cfg, args.redo_inference)

    if not args.skip_judge:
        judge_cfg = load_yaml(args.judge_config)
        if args.skip_interactive:
            logging.warning("You are skipping interactive review of LLM-as-judge outputs. The LLM-as-judge tends to have a high false positive rate, "
                            "often rejecting too many responses, so we highly recommend keeping `--skip-interactive` unset.")

        raw_probe = run_judge_phase(
            judged_path,
            judge_cfg,
            raw_probe,
            args.api_config,
            args.redo_judge or args.redo_inference,
            args.skip_interactive
        )

    steer_stats = main_steerability_evaluation(raw_probe, not args.skip_judge, uvicorn_cfg["goal_dimensions"])
    steer_stats = add_run_info_to_stats(cfg, judge_cfg, steer_stats)
    print_steerability_summary(steer_stats)
    safe_json_dump(steer_stats, final_metrics_path, indent=4)
 
