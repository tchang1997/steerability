import asyncio
from argparse import ArgumentParser
from datetime import datetime
from itertools import cycle, product
import json
import os
import requests
import time
import warnings

USE_UNSLOTH = os.environ.get("USE_UNSLOTH", False)

if USE_UNSLOTH:
    from unsloth import FastLanguageModel, PatchFastRL
    PatchFastRL("GRPO", FastLanguageModel) # first import

from accelerate.logging import get_logger
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
from trl import (
    GRPOConfig,
    GRPOTrainer, # until further notice we use a patched version
    ModelConfig,
    LogCompletionsCallback,
    get_peft_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from instruction_generator import get_instruction_generator
from rewards import (
    map_to_goalspace,
    REGISTRY,
    steerability_reward_wrapper,
    format_reward_func,
    english_only_reward_func,
)
from utils.train_config_dataclasses import (
    SteerabilityProbeConfig,
    GoalspaceServerConfig,
    UnslothLoraConfig,
    RewardConfig,
)

from beartype.typing import Callable, List, Tuple
from typing import Optional

logger = get_logger(__name__, log_level="INFO")

@beartype
def get_reward_funcs(
    reward_config: RewardConfig,
    model_name: str,
) -> Tuple[List[Callable], List[float]]:
    reward_func_names = reward_config.rewards
    reward_funcs = []
    for name in reward_func_names:
        if name in REGISTRY:
            reward_funcs.append(REGISTRY[name])
        else:
            warnings.warn(f"Reward function {name} is not registered in rewards.py; skipping. Valid rewards are: {list(REGISTRY.keys())}") 
    
    if model_name.split("/")[0] == "deepseek-ai":
        reward_funcs
        reward_funcs.append(REGISTRY["english_only"])

    baseline_weights = [1] * len(reward_funcs)
    eval_reward_names = reward_config.eval_rewards
    for name in eval_reward_names:
        if name in REGISTRY:
            reward_funcs.append(REGISTRY[name])
        else:
            warnings.warn(f"Reward function {name} is not registered in rewards.py; skipping. Valid rewards are: {list(REGISTRY.keys())}") 
    
    eval_weights = [0] * len(eval_reward_names) # weighting = 0 -> rewards will be logged during training but not optimized
    return reward_funcs, baseline_weights + eval_weights


@beartype
def prepare_steerability_probe(
        probe_config: SteerabilityProbeConfig, # TODO: generate a new sadness-surprise probe
        model_name: str,
    ) -> Tuple[Dataset, Dataset]: # TODO: make configurable
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

    # create eval dataset
    eval_probe = final_probe.iloc[:probe_config.num_prompts_for_eval] # non-random currently
    if probe_config.canary_file is not None:

        with open(probe_config.canary_file, "r", encoding="utf-8") as f:
            canaries = json.load(f)
        
        n_steering_goals = delta_goals.shape[1] # to use for sampling later on

        canary_deltas = []
        for col in delta_goals.columns:
            row_positive = {c: np.nan for c in delta_goals.columns}
            row_negative = {c: np.nan for c in delta_goals.columns}
            row_positive[col] = probe_config.canary_goal_magnitude
            row_negative[col] = -probe_config.canary_goal_magnitude
            canary_deltas.append(row_positive)
            canary_deltas.append(row_negative)
        canary_instructions = instruction_generator.sample_prompt(
            pd.DataFrame(canary_deltas, columns=delta_goals.columns), # uh. yeah. let's refactor later.
            None
        )
        canary_data = []
        for (canary, instruction), inst_deltas in zip(product(canaries, canary_instructions), cycle(canary_deltas)):
            prompt = instruction + "\n\n" + canary
            if probe_config.apply_conversational_format:
                prompt = [{
                    'role': 'user',
                    'content': prompt,
                }]
            base_dict = {
                "text": canary,
                "instructions": instruction,
                "prompt": prompt,
                "model_name": model_name,
            }
            base_dict.update(inst_deltas)
            canary_data.append(base_dict)
        canary_df = pd.DataFrame(canary_data)
        eval_probe = pd.concat([eval_probe, canary_df], axis=0, ignore_index=True)

    dataset = Dataset.from_pandas(final_probe)
    eval_dataset = Dataset.from_pandas(eval_probe)
    return dataset, eval_dataset

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
        model_args.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code, 
        revision=model_args.model_revision,
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

    if USE_UNSLOTH:
        parser = HfArgumentParser((
            GRPOConfig,
            ModelConfig,
            SteerabilityProbeConfig,
            GoalspaceServerConfig,
            RewardConfig,
            UnslothLoraConfig
        ))
        (
            training_config,
            model_config,
            probe_config,
            server_config,
            reward_config,
            unsloth_config, 
        ) = parser.parse_yaml_file(base_args.config)
    else:
        parser = HfArgumentParser((
            GRPOConfig,
            ModelConfig,
            SteerabilityProbeConfig,
            GoalspaceServerConfig,
            RewardConfig,
        ))
        (
            training_config,
            model_config,
            probe_config,
            server_config,
            reward_config,
        ) = parser.parse_yaml_file(base_args.config)


    print("Preparing data...")
    dataset, eval_dataset = prepare_steerability_probe(probe_config, model_config.model_name_or_path)
    test_server(server_config, dataset[probe_config.source_text_col][0])

    print("Preparing trainer...")
    reward_funcs = get_reward_funcs(model_config, model_config.model_name_or_path)
    #reward_funcs = [steerability_reward_wrapper]
    #if model_config.model_name_or_path.split("/")[0] == "deepseek-ai":
    #    reward_funcs += [format_reward_func, english_only_reward_func]
    print("Reward functions:")
    print(*[rf.__name__ for rf in reward_funcs], sep="\n")
    print()
    if reward_config.steering_goals is not None:
        print("Steering goals:")
        print(*reward_config.steering_goals, sep="\n")
        print()

    peft_config = get_peft_config(model_config) if not USE_UNSLOTH else None

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
        print(f"Loading {model_config.model_name_or_path} from revision {model_config.model_revision}")
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.model_revision,
        )   
        tokenizer = get_tokenizer(model_config)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_config,
        train_dataset=dataset,
        eval_dataset=eval_dataset, # generations can be done on the training data -- if it can't even overfit to training, how can it ever work on future eval datasets?
        peft_config=peft_config, # GRPOTrainer does the get_peft_model for us
        reward_extra_kwargs=vars(reward_config),
    )
    callbacks = [
        LogCompletionsCallback(
            trainer=trainer,
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
