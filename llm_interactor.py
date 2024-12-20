import json
import os
import re

from beartype import beartype
import numpy as np
import pandas as pd
from sammo.base import Template
from sammo.runners import OpenAIChat, MockedRunner
from sammo.components import Output, GenerateText
from sammo.throttler import AtMost

from custom_runners import DeepInfraChat
from goals import Goalspace
from instruction_generator import InstructionGenerator

from typing import Dict, Optional, Tuple, Union

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
        filtered = probe[(probe[source_norm] > 0) & (probe[source_norm] < 1)]
        min_pair = filtered.loc[filtered[source_norm].idxmin()]
        max_pair = filtered.loc[filtered[source_norm].idxmax()]
        x1, y1 = min_pair[goal], min_pair[source_norm]
        x2, y2 = max_pair[goal], max_pair[source_norm]
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1 # w.l.o.g.

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
        cache_file: str,
        api_config: str,
        timeout: Optional[int] = 120,
        max_simul_calls: Optional[int] = 3,
        max_tokens: Optional[int] = None,
        inst_context_delimiter: Optional[str] = "\n\n",
        text_gen_kwargs: Optional[Dict] = {},
    ):
        self.llm_name = llm_name
        self.chat_type = chat_type
        self.cache_file = cache_file
        self.instruction_generator = instruction_generator
        self.inst_context_delimiter = inst_context_delimiter
        self.text_gen_kwargs = text_gen_kwargs

        if os.path.isfile(api_config):
            with open(api_config, "r") as f:
                api_config = json.load(f)
                if re.search(r"[^\w\d.-]", llm_name):
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
        )

    @beartype
    def _get_sammo_runner(
        self,
        chat_type: str,
        **kwargs,
    ):
        if chat_type == "openai":
            self.chat_instance = OpenAIChat(**kwargs)
        elif chat_type == "deepinfra":
            self.chat_instance = DeepInfraChat(**kwargs)
        elif chat_type == "mockup":
            seed_data = pd.read_csv("./data/default_seed_data.csv", index_col=0)
            self.chat_instance = MockedRunner(seed_data["text"].tolist())
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
    ):
        return self.instruction_generator.sample_prompt(delta_goals, target_goals)


    @beartype
    def call_llm(
        self,
        prompts: Union[list, pd.Series],
        verbose: Optional[bool] = False,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]
        outputs = Output(GenerateText(Template("{{input}}"), **self.text_gen_kwargs)).run(self.chat_instance, prompts.tolist())
        final_output = []
        for raw_resp in outputs.outputs.llm_responses:
            clean_resp = self.instruction_generator.clean_response(raw_resp)
            try:
                final_output.append(clean_resp)
            except Exception as e:
                import traceback
                print("Raised exception:")
                print(traceback.format_exc())
        if verbose:
            print("LLM response:", final_output)
        return pd.Series(final_output, name="llm_response")


    @beartype
    def generate_steerability_data(
        self,
        probe: pd.DataFrame,
        prompts: Union[list, pd.Series],
        verbose: Optional[bool] = False,
    ):
        if not isinstance(prompts, pd.Series):
            prompts = pd.Series(prompts, name="instruction")
        raw_inputs = prompts.str.cat(probe["text"], sep=self.inst_context_delimiter)
        llm_outputs = self.call_llm(raw_inputs, verbose=verbose)
        goalspace = Goalspace.create_default_goalspace_from_probe(probe)
        goalspace_out = goalspace(llm_outputs.tolist(), return_pandas=True).add_prefix("output_raw_")
        out_normed = renormalize_goalspace(probe, goalspace_out)

        steerability_data = pd.concat([
            probe,
            prompts,
            raw_inputs,
            pd.Series(llm_outputs, name="raw_outputs"),
            goalspace_out,
            out_normed,
        ], axis=1)

        # this creates a "signature" that helps disambiguate experiments
        steerability_data["instruction_generator"] = self.instruction_generator.__class__.__name__
        steerability_data["llm"] = self.llm_name
        steerability_data["chat_type"] = self.chat_type
        steerability_data["endpoint"] = self.url_endpoint
        return steerability_data
