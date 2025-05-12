from collections import defaultdict
import json
import os
from pathlib import Path
import re

from aiohttp import ClientSession
import asyncio
from beartype import beartype
import numpy as np
import pandas as pd
from ruamel.yaml import YAML
from sammo.base import Template
from sammo.runners import OpenAIChat, MockedRunner
from sammo.components import Output, GenerateText
from sammo.throttler import AtMost
from tqdm.auto import tqdm

from steerability.custom_runners import VLLMOpenAIChat
from steerability.goals import Goalspace
from steerability.instruction_generator import InstructionGenerator
from steerability.rewards import send_request
from steerability.utils.model_output_cleaner import clean_model_output
from steerability.utils.probe_utils import get_nonnull_targets

from beartype.typing import Dict
from typing import Optional, Union

import logging
logger = logging.getLogger(__name__)

DEFAULT_LLM_CALL_TIMEOUT = 3600

@beartype
def renormalize_goalspace(
        probe: pd.DataFrame,
        raw_out: pd.DataFrame,
    ):
    # the probe df contains the un-normalized + normalized goal-space mappings of the original texts. We can reverse-engineer and apply the same mappings to the raw output s. Assumes min-max zero-one scaling.
    goal_names = [col.split("source_", 1)[1] for col in probe.filter(like="source_", axis=1)] 
    normalized = []
    for goal in goal_names:
        source_norm = f"source_{goal}"
        if f"output_raw_{goal}" not in raw_out.columns:
            continue

        filtered = probe[(probe[source_norm] > 0) & (probe[source_norm] < 1)]
        min_pair = filtered.loc[filtered[source_norm].idxmin()]
        max_pair = filtered.loc[filtered[source_norm].idxmax()]
        x1, y1 = min_pair[goal], min_pair[source_norm]
        x2, y2 = max_pair[goal], max_pair[source_norm]
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1 # w.l.o.g.
        x_min_pred = -intercept / slope
        x_max_pred = (1 - intercept) / slope
        print(f"Recovered goal range for goal `{goal}`: [{x_min_pred:.2f}, {x_max_pred:2f}]")
        norm_col = pd.Series(np.clip(raw_out[f"output_raw_{goal}"] * slope + intercept, 0, 1), name=f"output_{goal}")
        normalized.append(norm_col)
    return pd.concat(normalized, axis=1)


class LLMInteractor(object):

    @beartype
    def __init__(
        self,
        instruction_generator: InstructionGenerator,
        llm_name: str,
        chat_type: str,
        cache_file: str | Path,
        api_config: str | Path,
        timeout: Optional[int] = 120,
        max_simul_calls: Optional[int] = 3,
        max_tokens: Optional[int] = None,
        inst_context_delimiter: Optional[str] = "\n\n",
        text_gen_kwargs: Optional[Dict] = {},
        port: Optional[int] = None,
        async_mode: Optional[bool] = False,
        goalspace_port: Optional[int] = 16641, # 129^2, since 16384 is taken
        max_simul_goalspace_reqs: Optional[int] = 24,
    ):
        self.llm_name = llm_name
        self.chat_type = chat_type
        self.cache_file = cache_file
        self.instruction_generator = instruction_generator
        self.inst_context_delimiter = inst_context_delimiter
        self.num_generations = text_gen_kwargs.get("num_generations", 1)
        self.text_gen_kwargs = text_gen_kwargs

        self.async_mode = async_mode
        self.goalspace_port = goalspace_port
        self.max_simul_goalspace_reqs = max_simul_goalspace_reqs
        logger.info(f"Creating LLMInteractor with args: llm_name={self.llm_name}, chat_type={self.chat_type}, cache_file={self.cache_file}, "
                f"instruction_generator={self.instruction_generator.__class__.__name__}, num_generations={self.num_generations}, "
                f"text_gen_kwargs={self.text_gen_kwargs}")

        if os.path.isfile(api_config):
            with open(api_config, "r") as f:
                api_config = json.load(f)
                if re.search(r"[^\w\d.\-\+]", llm_name):
                    if "/" in llm_name:
                        logger.info("`/` detected in model name. Checking HuggingFace to ensure model exists.")
                        from huggingface_hub import HfApi
                        api = HfApi()
                        api.model_info(llm_name)
                    else:
                        raise ValueError("Forbidden characters in LLM name. Must be alphanumeric or `-`.") 
        else:
            if chat_type != "mockup":
                raise ValueError(f"API config {api_config} not found. Check paths.")

        self.chat_instance = self._get_sammo_runner(
            chat_type,
            model_id=llm_name,
            api_config=api_config, 
            cache=os.path.join("./cache", cache_file),
            timeout=timeout,
            rate_limit=AtMost(max_simul_calls, "running"),
            max_context_window=max_tokens,
            port=port,
        )

    @classmethod
    def from_configs(
        cls,
        instruction_generator,
        chat_type: str,
        api_config: str,
        cfg: dict,
        vllm_cfg: dict,
        uvicorn_cfg: dict,
    ):
        return cls(
            instruction_generator,
            cfg["model_id"],
            chat_type,
            api_config=api_config,
            cache_file=cfg["model_id"].replace("/", "__") + ".tsv",
            timeout=DEFAULT_LLM_CALL_TIMEOUT,
            async_mode=True,
            port=vllm_cfg["port"],
            goalspace_port=uvicorn_cfg["port"],
            max_tokens=cfg["max_tokens"],
            max_simul_calls=cfg["rate_limit"],
            max_simul_goalspace_reqs=uvicorn_cfg["workers"],
            text_gen_kwargs=cfg["text_gen_kwargs"],
        )

    @beartype
    def _get_sammo_runner(
        self,
        chat_type: str,
        port: Optional[int] = None,
        **kwargs,
    ):
        if chat_type == "openai":
            self.chat_instance = OpenAIChat(**kwargs)
        elif chat_type == "vllm":
            self.chat_instance = VLLMOpenAIChat(port=port, **kwargs)
        else:
            raise ValueError(f"Chat type `{chat_type}` is not recognized.")
        try:
            self.url_endpoint = self.chat_instance._rest_url()
        except AttributeError:
            self.url_endpoint = None
        return self.chat_instance

    @beartype
    def write_prompts(
        self,
        delta_goals: pd.DataFrame,
        target_goals: pd.DataFrame,
        **kwargs
    ):
        return self.instruction_generator.sample_prompt(delta_goals, target_goals, **kwargs)


    @beartype
    def call_llm(
        self,
        prompts: Union[list, pd.Series],
        verbose: Optional[bool] = False,
    ) -> pd.DataFrame:
        if isinstance(prompts, str):
            prompts = [prompts]
        # since we do caching -- we can later check if all outputs are coherent...
        outputs = Output(GenerateText(Template("{{input}}"), **self.text_gen_kwargs)).run(self.chat_instance, prompts.tolist())

        final_output = []
        raw_output = []
        for i, raw_resp in enumerate(outputs.outputs.llm_responses): 
            try:
                iter_obj = raw_resp if isinstance(raw_resp[0], str) else raw_resp[0]
                for resp in iter_obj: 
                    clean_resp = clean_model_output(self.llm_name, resp) # by default, only return one response
                    clean_resp = self.instruction_generator.clean_response(clean_resp) 
                    raw_output.append(resp)
                    final_output.append(clean_resp)
            except Exception as e:
                if outputs.outputs[i].value == "KeyError('choices')":
                    print("Forcibly returned exception detected, causing a KeyError when post-processing. This usually happens due to a cached timeout error or content policy violation. Proceeding with null string.")
                    raw_output.append("Error: content policy violation detected!") # HACK 
                    final_output.append("Error: content policy violation detected!")
                    continue
                import traceback
                print("Unhandled exception raised during LLM response post-processing. This can happen if the server fails to respond with the expected format. "
                      "This is usually resolved by retrying the prompt. Please re-run the current script manually to redo those calls, and previous calls will be fetched from the cache.")
                print("Full traceback:")
                print(traceback.format_exc())
                raise e
        if verbose:
            print("LLM response:", final_output)
        return pd.DataFrame({
            "raw_response": raw_output,
            "llm_response": final_output,
        })

    async def safe_request(self, i, session, text, goals, port):
        try:
            if not isinstance(text, list):
                text = [text]
            result = await send_request(session, text, goals=goals, port=port, normalize=False) # raw outputs at this stage only!
            return i, result
        except Exception as e:
            print("An exception was raised when sending request", i)
            return i, e 

    @beartype 
    async def get_goalspace_mappings(self, probe: pd.DataFrame, responses: list[str], max_for_debug: Optional[int] = None, batch_size: Optional[int] = 32):
        goalspace = Goalspace.create_default_goalspace_from_probe(probe)
        goals = goalspace.get_goal_names(snake_case=True)
        if max_for_debug is not None:
            responses = responses[:max_for_debug]

        # chunk into batches
        resps_batched = [responses[i:i + batch_size] for i in range(0, len(responses), batch_size)]

        if self.async_mode:
            results = [None] * len(resps_batched)
            sem = asyncio.Semaphore(self.max_simul_goalspace_reqs)  # define this in __init__ or pass in

            async def limited_safe_request(i, session, text):
                async with sem:
                    return await self.safe_request(i, session, text, goals=goals, port=self.goalspace_port)

            async with ClientSession() as session:
                tasks = [limited_safe_request(i, session, text) for i, text in enumerate(resps_batched)]

                for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                    i, response = await fut
                    results[i] = response # dict[str -> list[float]]

            # handle lists of lists in future
            combined = defaultdict(list)
            for d in results:
                for k, v in d.items(): # str -> list[float]
                    combined[k].extend(v)
            mappings = pd.DataFrame(combined)
        else:
            mappings = goalspace(responses, return_pandas=True)
        return mappings
    
    def generate_prompts(
        self,
        probe: pd.DataFrame,
        **kwargs
    ):
        delta_goals = probe.filter(like='delta_', axis=1)
        target_corrected = get_nonnull_targets(probe)

        prompts = self.write_prompts(
            delta_goals,
            target_corrected, 
            **kwargs, # cfg.get("inst_addons", {})
        ) 
        print("EXAMPLE PROMPT:")
        print(f"---\n{prompts[0]}---\n") 
        return prompts

    @beartype
    def generate_steerability_data(
        self,
        probe: pd.DataFrame,
        prompts: Union[list, pd.Series],
        seed_data: Optional[pd.DataFrame] = None,
        normalization_cfg: Optional[str | Path] = None, 
        verbose: Optional[bool] = False,
        max_for_debug: Optional[int] = None,
    ):
        if not isinstance(prompts, pd.Series):
            prompts = pd.Series(prompts, name="instruction")
        raw_inputs = prompts.str.cat(probe["text"], sep=self.inst_context_delimiter)
        llm_outputs = self.call_llm(raw_inputs, verbose=verbose)
        # TODO: implement retry -- given caching, should be safe to just call self.call_llm again

        goalspace_out = asyncio.run(self.get_goalspace_mappings(probe, llm_outputs["llm_response"].tolist(), max_for_debug=max_for_debug)).add_prefix("output_raw_") # TODO: now that we just have a goalspace mapping server...perhaps we make the server configurable, and asyncio.run this?
        
        if normalization_cfg is not None:
            with open(normalization_cfg, "r") as f:
                norm = YAML(typ="safe").load(normalization_cfg)
            norm_cols = []
            for goal in norm.keys():
                if f"output_raw_{goal}" in goalspace_out.columns:
                    goal_min, goal_max = norm[goal]["min"], norm[goal]["max"]
                    norm_col = pd.Series(np.clip((goalspace_out[f"output_raw_{goal}"] - goal_min) /  (goal_max - goal_min), 0, 1), name=f"output_{goal}")
                    norm_cols.append(norm_col)
            out_normed = pd.concat(norm_cols, axis=1)
        elif seed_data is not None:
            out_normed = renormalize_goalspace(seed_data, goalspace_out)
        else:
            raise ValueError("Cannot normalize goal-space outputs without seed data for reference, or a normalization config.")

        if self.num_generations == 1:
            steerability_data = pd.concat([
                probe,
                prompts,
                raw_inputs,
                llm_outputs,
                goalspace_out,
                out_normed,
            ], axis=1)
        else:
            steerability_data = pd.concat([
                probe.loc[np.repeat(probe.index, self.num_generations)].reset_index(drop=True),
                prompts.loc[np.repeat(prompts.index, self.num_generations)].reset_index(drop=True),
                raw_inputs,
                llm_outputs,
                goalspace_out,
                out_normed,
            ], axis=1)

        # this creates a "signature" that helps disambiguate experiments
        steerability_data["instruction_generator"] = self.instruction_generator.__class__.__name__
        steerability_data["llm"] = self.llm_name
        steerability_data["chat_type"] = self.chat_type
        steerability_data["endpoint"] = self.url_endpoint
        return steerability_data
