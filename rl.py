import asyncio
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
import math
import os
import re
import requests
import time

USE_UNSLOTH = os.environ.get("USE_UNSLOTH", False)

if USE_UNSLOTH:
    from unsloth import FastLanguageModel, PatchFastRL
    PatchFastRL("GRPO", FastLanguageModel) # first import

from accelerate.logging import get_logger
from aiohttp import ClientSession
from beartype import beartype
from datasets import Dataset
import numpy as np
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    GenerationConfig,
)
import trl
from trl import (
    GRPOConfig,
    GRPOTrainer, # until further notice we use a patched version
    ModelConfig,
    LogCompletionsCallback,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from grpo_trainer_patch import GRPOTrainerWithPatch
from instruction_generator import get_instruction_generator
from utils.train_config_dataclasses import SteerabilityProbeConfig, GoalspaceServerConfig, UnslothLoraConfig
from utils.model_output_cleaner import clean_model_output, is_chat_completion_format

from beartype.typing import List
from typing import Optional, Union

logger = get_logger(__name__, log_level="INFO")
CHINESE_PATTERN = re.compile(r"[\u4E00-\u9FFF]")

@beartype
def prepare_steerability_probe(
        probe_config: SteerabilityProbeConfig,
        model_name: str,
    ) -> Dataset: # TODO: make configurable
    probe = pd.read_csv(probe_config.steerability_probe, index_col=0) # currently, we're using a static, pre-generated probe -- probably good to save goalspace mapping time
    source_text_idx = probe[probe_config.source_text_id_col] \
        .drop_duplicates() \
        .sample(n=probe_config.n_source_texts, random_state=probe_config.probe_sampling_seed)
    print("Texts selected:", source_text_idx.tolist())
    probe_subset = probe[probe[probe_config.source_text_id_col].isin(source_text_idx)]

    final_probe = probe_subset.groupby(probe_config.source_text_id_col, group_keys=False).apply(
        lambda x: x.sample(
            n=min(len(x), probe_config.instructions_per_text),
            random_state=probe_config.probe_sampling_seed
        )
    ).reset_index(drop=True)

    instruction_generator = get_instruction_generator(probe_config.prompt_strategy)
    delta_goals = final_probe.filter(like='delta_', axis=1) # This is heavily reliant on how the probe is implemented at an earlier stage -- target for further refactor
    target_goals = final_probe.filter(like='target_', axis=1)
    source_goals = final_probe.filter(like="source_", axis=1)

    for target_col in target_goals.columns:
        source_col = target_col.replace("target_", "source_")
        final_probe[target_col] = target_goals[target_col].fillna(source_goals[source_col]) # patch the probe 

    final_probe["instructions"] = instruction_generator.sample_prompt(delta_goals, target_goals)
    final_probe["prompt"] = final_probe["instructions"] + "\n\n" + final_probe[probe_config.source_text_col]
    if probe_config.apply_conversational_format:
        final_probe["prompt"] = final_probe["prompt"].apply(
            lambda content: [{
                'role': 'user',
                'content': content,
            }]
        )
    final_probe["model_name"] = model_name # ...I don't want to talk about this HACK
    print("Final probe length:", len(final_probe))
    print("Columns:", final_probe.columns)

    dataset = Dataset.from_pandas(final_probe)
    return dataset

def await_server(port: Optional[int] = 12121, timeout: Optional[int] = 300):
    url = f"http://127.0.0.1:{port}/health"
    print(f"⏳ Waiting for server at {url} to be ready...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200 and response.json().get("ready", False):
                print("✅ Goalspace-mapping server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{timestamp}: Waiting for server... (make sure you've run `uvicorn goalspace_server:app --host 127.0.0.1 --port {port} --workers [NUM_WORKERS]`)")
        time.sleep(5)
    raise RuntimeError(f"❌ Goalspace-mapping server did not start within the time limit ({timeout} seconds).")

async def send_request(
        session: ClientSession,
        texts: List[str],
        goals: Optional[List[str]] = None,
        port: Optional[int] = 12121
    ):
    inference_url = f"http://127.0.0.1:{port}/goalspace" # why stop here? We can even throw vLLM in here someday
    payload = {"texts": texts}
    if goals is not None:
        payload["goals"] = goals
    async with session.post(inference_url, json=payload) as response:
        return await response.json()

async def map_to_goalspace(
        texts: Union[str, List[str]],
        goals: Optional[List[str]] = None,
        port: Optional[int] = 12121,
        n_workers: Optional[int] = 4
    ):
    if isinstance(texts, str):
        texts = [texts]
    batch_size = math.ceil(len(texts) / n_workers)
    text_batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    async with ClientSession() as session:
        tasks = [send_request(session, batch, goals=goals, port=port) for batch in text_batches]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
    merged_responses = defaultdict(list)
    for response in responses:
        for goal, mapping_values in response.items(): 
            merged_responses[goal].extend(mapping_values)
    return merged_responses
    
@beartype
def steerability_reward_wrapper(completions, **kwargs) -> Union[List[float], np.ndarray]: # TODO: ortho penalty, miscalibration, variance reg.?
    """
        In order to allow for *literally anything* to be used as a goal-space mapper, we delegate goalspace-mapping to an 
        external server and ping it with asyncio. This introduces some I/O overhead, yes, but `accelerate launch` + DeepSpeed ZeRO gets
        very confused and frustrated when I pass in a wrapper class that includes other model wrapper classes that cache inputs/outputs...
        goal-space mappings were not initially designed to be used directly as a reward model, so this is a happy medium (I think).
    """
    if is_chat_completion_format(completions):
        completions = [completion[0]["content"] for completion in completions]
    cleaned_completions = [clean_model_output(kwargs["model_name"][0], completion) for completion in completions]
    mappings = asyncio.run(map_to_goalspace(cleaned_completions)) # this is z-hat
    macro_negreward = np.zeros(len(cleaned_completions))
    for goal, values in mappings.items():
        macro_negreward += np.square(np.array(kwargs[f"target_{goal}"]) - np.array(values)) # add (\hat{z}_i - z*_i)^2 -> squared L2. Should throw a shape error if any goals are missing.
    rewards = (-np.sqrt(macro_negreward / len(mappings)) + 1) # normalize to [0, 1]; macro_negreward is in range [-sqrt(len(mappings), 0]
    return rewards

def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    if is_chat_completion_format(completions):
        completions = [completion[0]["content"] for completion in completions]
    pattern = r"^<think>.*?</think>" # keep the deepseek format
    matches = [re.match(pattern, content) for content in completions]
    return [1.0 if match else 0.0 for match in matches]

def english_only_reward_func(completions, **kwargs):
    """most common language mixing is Chinese, so just detect + penalize"""
    if is_chat_completion_format(completions):
        completions = [completion[0]["content"] for completion in completions]
    return [0.0 if CHINESE_PATTERN.search(content) else 1.0 for content in completions]

def test_server(server_config: GoalspaceServerConfig, testing_text: str):
    print("Testing goalspace mapping server...")
    print("SOURCE:")
    print("---")
    print(testing_text)
    await_server(server_config.port, server_config.startup_timeout)
    test_mapping = asyncio.run(map_to_goalspace(testing_text))
    print("MAPPING:")
    print(test_mapping)
    print("---\n")

def get_tokenizer(model_args):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.pad_token = tokenizer.eos_token 
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    return tokenizer

if __name__ == '__main__':
    # Implementation based on https://github.com/huggingface/trl/blob/main/examples/scripts/ppo/ppo.py with some changes.
    base_parser = ArgumentParser()
    base_parser.add_argument("--config", required=True)
    base_args, *_ = base_parser.parse_known_args()

    parser = HfArgumentParser((
        GRPOConfig,
        ModelConfig,
        SteerabilityProbeConfig,
        GoalspaceServerConfig,
        UnslothLoraConfig
    ))
    training_config, model_config, probe_config, server_config, unsloth_config = parser.parse_yaml_file(base_args.config)

    print("Preparing data...")
    dataset = prepare_steerability_probe(probe_config, model_config.model_name_or_path)
    test_server(server_config, dataset[probe_config.source_text_col][0])

    print("Preparing trainer...")
    reward_funcs = [steerability_reward_wrapper]
    if model_config.model_name_or_path.split("/")[0] == "deepseek-ai":
        reward_funcs += [format_reward_func, english_only_reward_func]

    if USE_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name =model_config.model_name_or_path,
            max_seq_length=training_config.max_prompt_length + training_config.max_completion_length,
            load_in_4bit=model_config.load_in_4bit, 
            fast_inference=training_config.use_vllm, # Enable vLLM fast inference
            max_lora_rank=model_config.lora_r,
            gpu_memory_utilization=training_config.vllm_gpu_memory_utilization, # Reduce if out of memory
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=model_config.lora_r, 
            target_modules=model_config.lora_target_modules,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
            use_gradient_checkpointing=unsloth_config.unsloth_grad_checkpointing, 
            random_state=unsloth_config.unsloth_random_state,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path)
        tokenizer = get_tokenizer(model_config)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_config,
        train_dataset=dataset,
        eval_dataset=dataset, # generations can be done on the training data -- if it can't even overfit to training, how can it ever work on future eval datasets?
    )
    callbacks = [
        LogCompletionsCallback(
            trainer=trainer,
            num_prompts=8,
            freq=1,
            generation_config=GenerationConfig(
                max_new_tokens=training_config.max_completion_length,
                do_sample=False, # implicitly temp = 0
            )
        )
    ]
    for callback in callbacks:
        trainer.add_callback(callback)
    trainer.train() 
    trainer.save_model(training_config.output_dir)

    if USE_UNSLOTH:
        model.save_lora(os.path.join(training_config.output_dir, unsloth_config.lora_adapter_name))
