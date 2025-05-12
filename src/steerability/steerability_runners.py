from pathlib import Path
from string import Template as StringTemplate

import pandas as pd

from steerability.instruction_generator import load_instruction_database, get_instruction_generator
from steerability.llm_interactor import LLMInteractor
from steerability.llm_launcher import block_until_healthy, launch_goalspace_server, launch_judge
from steerability.utils.llm_groundedness import create_problem_set, initialize_chat_instance, parse_and_score, interactive_review, judge

from typing import Any, Optional

import logging
logger = logging.getLogger(__name__)

NORMALIZATION_CFG = Path("./config/goalspace_defaults/normalization.yml")

def launch_steerability_eval(
        probe: pd.DataFrame,
        chat_type: str,
        api_config: str,
        cfg: dict[str, Any],
        vllm_cfg: dict[str, Any],
        uvicorn_cfg: dict[str, Any],
    ):
    prompt_strategy = cfg["prompt_strategy"]

    database = None
    if "instruction_path" in cfg:
        database = load_instruction_database(cfg["instruction_path"])
    instruction_generator = get_instruction_generator(
        prompt_strategy,
        database=database, # extracted via LLM from CoT traces
        prompter_kwargs=cfg.get("prompter_kwargs", {})
    )
    llm_interactor = LLMInteractor.from_configs(
        instruction_generator,
        chat_type,
        api_config,
        cfg,
        vllm_cfg,
        uvicorn_cfg
    )

    goalspace_proc = None
    exception = None
    try:
        goalspace_proc = launch_goalspace_server(uvicorn_cfg) # non-blocking

        prompts = llm_interactor.generate_prompts(probe, **cfg.get("inst_addons", {}))
        logger.info(f"Pinging endpoint: {llm_interactor.url_endpoint}")
        outputs = llm_interactor.generate_steerability_data(
            probe,
            prompts,
            normalization_cfg=NORMALIZATION_CFG, # one day let's not hardcode this
        ) # seed_data for normalization only -- refactor someday?
        block_until_healthy(goalspace_proc, uvicorn_cfg["port"]) # just in case
    except Exception as e:
        exception = e
    finally:
        if goalspace_proc is not None:
            if goalspace_proc.poll() is None:
                goalspace_proc.terminate()
                logger.info("Sent SIGTERM to goalspace server.")
            else:

                logger.debug("Goalspace server terminated gracefully.")
        if exception is not None:
            raise exception
    return outputs


def evaluate_and_review(pset: pd.DataFrame, judge_cfg: dict, api_config: dict, skip_interactive: Optional[bool] = False): # TODO -- one day, make this the main usage 
    chat_instance = initialize_chat_instance(
        judge_cfg["vllm_args"]["port"],
        api_config=api_config,
        cache_suffix="_groundedness_judge.tsv"
    )
    pset_answered = judge(pset, chat_instance)
    judged = parse_and_score(pset_answered)
    judged['rationale_approved'] = True
    judged['spot_check'] = False 
    
    if not skip_interactive: # for the full set-and-forget experience. But I don't dare expose this to CLI yet...
        judged = interactive_review(judged, judge_cfg["spot_check_size"])
    return judged

def run_interactive_llm_as_judge(judge_cfg: str, probe: pd.DataFrame, api_config: str, skip_interactive: Optional[bool] = False):
    exception = None
    judge_proc = None
    try:
        judge_proc = launch_judge(judge_cfg)
        with open(judge_cfg["prompt_file"], "r") as f:
            prompt_template = StringTemplate(f.read().strip())
        judge_port = judge_cfg["vllm_args"]["port"]
        pset = create_problem_set(
            probe,
            prompt_template,
            judge_cfg["source_col"],
            judge_cfg["response_col"],
            judge_cfg["seed"],
            port=judge_port,
        )
        reviewed = evaluate_and_review(pset, judge_cfg, api_config, skip_interactive=skip_interactive)
    except Exception as e:
        exception = e
    finally:
        if judge_proc is not None:
            if judge_proc.poll() is None:
                judge_proc.terminate() 
                logger.info("SIGTERM sent to judge vLLM instance.")
            else:
                logger.debug("Judge terminated gracefully.")
        if exception is not None:
            raise exception
    return reviewed
