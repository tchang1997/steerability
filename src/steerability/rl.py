import asyncio
from argparse import ArgumentParser
import ast
from datetime import datetime
import os
import requests
import time
import warnings

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

from steerability.instruction_generator import get_instruction_generator
from steerability.rewards import (
    map_to_goalspace,
    REGISTRY,
)
from steerability.utils.train_config_dataclasses import (
    SteerabilityProbeConfig,
    GoalspaceServerConfig,
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

    baseline_weights = [1.] * len(reward_funcs)
    eval_reward_names = reward_config.eval_rewards
    for name in eval_reward_names:
        if name in REGISTRY:
            reward_funcs.append(REGISTRY[name])
        else:
            warnings.warn(f"Reward function {name} is not registered in rewards.py; skipping. Valid rewards are: {list(REGISTRY.keys())}") 
    
    eval_weights = [0.] * len(eval_reward_names) # weighting = 0 -> rewards will be logged during training but not optimized
    return reward_funcs, baseline_weights + eval_weights

@beartype
def generate_instruction_columns(final_probe: pd.DataFrame, probe_config: SteerabilityProbeConfig, model_name: str) -> pd.DataFrame:
    instruction_generator = get_instruction_generator(probe_config.prompt_strategy)
    delta_goals = final_probe.filter(like='delta_', axis=1) # This is heavily reliant on how the probe is implemented at an earlier stage -- target for further refactor
    target_goals = final_probe.filter(like='target_', axis=1)
    source_goals = final_probe.filter(like="source_", axis=1)

    for target_col in target_goals.columns:
        source_col = target_col.replace("target_", "source_")
        final_probe[target_col] = target_goals[target_col].fillna(source_goals[source_col]) # patch the probe 

    final_probe["instructions"] = instruction_generator.sample_prompt(delta_goals, target_goals, disambig=True) 
    final_probe["prompt"] = final_probe["instructions"] + "\n\n" + final_probe[probe_config.source_text_col]
    if probe_config.apply_conversational_format:
        final_probe["prompt"] = final_probe["prompt"].apply(
            lambda content: [{
                'role': 'user',
                'content': content,
            }]
        )
    final_probe["model_name"] = model_name # so we can post-process reasoning traces via model_output_cleaner 
    return final_probe

@beartype
def create_cross_probe(source_text_df: pd.DataFrame, inst_df: pd.DataFrame):
    cross_probe_size = min(len(source_text_df), len(inst_df))
    cross_probe = source_text_df.copy().iloc[:cross_probe_size]

    delta_goals = cross_probe.filter(like='delta_', axis=1) # This is heavily reliant on how the probe is implemented at an earlier stage -- target for further refactor
    target_goals = cross_probe.filter(like='target_', axis=1)
    source_goals = cross_probe.filter(like="source_", axis=1)

    columns_to_direct_copy = delta_goals.columns.tolist() + ["instructions", "prompt"]
    cross_probe[columns_to_direct_copy] = inst_df[columns_to_direct_copy].iloc[:cross_probe_size].values # copy deltas from test set over 
    cross_probe[target_goals.columns] = (
        cross_probe[source_goals.columns].values[:cross_probe_size]
        + cross_probe[delta_goals.columns].values[:cross_probe_size] 
    ) # recompute z*
    for target_col in target_goals.columns:
        source_col = target_col.replace("target_", "source_")
        cross_probe[target_col] = cross_probe[target_col].fillna(cross_probe[source_col])
        cross_probe[target_col] = cross_probe[target_col].clip(0, 1) # patch the probe and ensure feasibility 
    # deltas are kept as-is for record-keeping (i.e., they may ``overflow''); but only z* is used for reward eval
    return cross_probe

@beartype
def create_train_val_split(probe: pd.DataFrame, probe_config: SteerabilityProbeConfig, model_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_probe = probe.sample(frac=0.5, random_state=probe_config.probe_sampling_seed)
    test_probe = probe.drop(train_probe.index)

    if probe_config.train_probe_path is not None:
        train_data = pd.read_csv(probe_config.train_probe_path, index_col=0)
        train_data["prompt"] = train_data["prompt"].apply(ast.literal_eval)
    else:
        train_source_texts = train_probe[probe_config.source_text_id_col] \
            .drop_duplicates() \
            .sample(
                n=probe_config.n_source_texts,
                random_state=probe_config.train_source_sampling_seed,
                replace=False
            ).tolist() # sampling w/o replacement may induce some bias. No sample-weighting; weights already precalculated.     
        print("Train texts:", train_source_texts)

        train_subset = train_probe[train_probe[probe_config.source_text_id_col].isin(train_source_texts)]
        train_data = train_subset.groupby(probe_config.source_text_id_col, group_keys=False).apply(
            lambda x: x.sample(
                n=min(len(x), probe_config.instructions_per_text),
                random_state=probe_config.train_inst_sampling_seed,
                replace=False,
            )
        ).reset_index(drop=True) # this'll sample the same instruction indices per source text, which shouldn't be the same as the same instructions 
        train_data = generate_instruction_columns(train_data, probe_config, model_name)

    if probe_config.test_probe_path is not None:
        test_data = pd.read_csv(probe_config.test_probe_path, index_col=0)
        test_data["prompt"] = test_data["prompt"].apply(ast.literal_eval)
    else:
        test_source_texts = test_probe[probe_config.source_text_id_col] \
            .drop_duplicates() \
            .sample(
                n=probe_config.num_test_prompts_for_eval // probe_config.insts_per_probe_source_text, 
                random_state=probe_config.test_source_sampling_seed,
                replace=False,
            ).tolist()
        print("Test texts:", test_source_texts)
        test_subset = test_probe[test_probe[probe_config.source_text_id_col].isin(test_source_texts)]
        test_data = test_subset.groupby(probe_config.source_text_id_col, group_keys=False).apply(
            lambda x: x.sample(
                n=min(len(x), probe_config.insts_per_probe_source_text),
                random_state=probe_config.test_inst_sampling_seed,
                replace=False,
            )
        ).reset_index(drop=True) # this'll sample the same instruction indices per source text, which shouldn't be the same as the same instructions 
        test_data = generate_instruction_columns(test_data, probe_config, model_name)
    return train_data, test_data

@beartype
def prepare_steerability_probe(
        probe_config: SteerabilityProbeConfig, 
        model_name: str,
        run_name: str,
        training_probe_dir: Optional[str] = "./training_probes/",
    ) -> Tuple[Dataset, Dataset]: # TODO: make configurable
    assert probe_config.num_train_prompts_for_eval % probe_config.instructions_per_text == 0
    assert probe_config.num_test_prompts_for_eval % probe_config.instructions_per_text == 0
 
    probe = pd.read_csv(probe_config.steerability_probe, index_col=0) # currently, we're using a static, pre-generated probe -- probably good to save goalspace mapping time
    train_data, test_data = create_train_val_split(probe, probe_config, model_name)

    print("Final probe length:", len(train_data))
    print("Columns:", train_data.columns)
    min_wt, max_wt = train_data["sampling_weights_mean"].min(), train_data["sampling_weights_mean"].max()
    train_data["sampling_weights_mean"] = train_data["sampling_weights_mean"].clip(probe_config.clip_min, probe_config.clip_max)
    print(f"Sampling weights range from ({min_wt}, {max_wt})")

    # create tracking "cross-probes"
    test_eval_subset = test_data 
    if probe_config.insts_per_probe_source_text is None:
        train_eval_subset = train_data.iloc[:probe_config.num_train_prompts_for_eval]
    else:
        assert probe_config.num_train_prompts_for_eval % probe_config.insts_per_probe_source_text == 0 
        assert probe_config.num_test_prompts_for_eval % probe_config.insts_per_probe_source_text == 0 
        train_groups_needed = probe_config.num_train_prompts_for_eval // probe_config.insts_per_probe_source_text
        first_train_groups = train_data[probe_config.source_text_id_col].unique()[:train_groups_needed]
        train_eval_superset = train_data[train_data[probe_config.source_text_id_col].isin(first_train_groups)]
        train_eval_subset = train_eval_superset.groupby(probe_config.source_text_id_col).head(probe_config.insts_per_probe_source_text)
    if probe_config.cross_probes:
        train_source_test_inst = create_cross_probe(train_eval_subset, test_eval_subset)
        test_source_train_inst = create_cross_probe(test_eval_subset, train_eval_subset)

        assert (test_eval_subset.filter(regex="^(original_|text|source*)", axis=1).values == test_source_train_inst.filter(regex="^(original_|text|source*)", axis=1).values).all()
        assert (train_eval_subset.filter(regex="^(original_|text|source*)", axis=1).values == train_source_test_inst.filter(regex="^(original_|text|source*)", axis=1).values).all()
        assert (test_eval_subset.filter(regex="^(delta_|prompt|instructions)", axis=1).fillna(-1).values == train_source_test_inst.filter(regex="^(delta_|prompt|instructions)", axis=1).fillna(-1).values).all()
        assert (train_eval_subset.filter(regex="^(delta_|prompt|instructions)", axis=1).fillna(-1).values == test_source_train_inst.filter(regex="^(delta_|prompt|instructions)", axis=1).fillna(-1).values).all()
        probes_to_combine = [
            train_eval_subset,
            train_source_test_inst,
            test_source_train_inst, 
            test_eval_subset,
        ]
        split_names = ["train", "train_source_test_inst", "test_source_train_inst", "test"]
    else:
        probes_to_combine = [train_eval_subset, test_eval_subset]
        split_names = ["train", "test"]
    # create eval dataset
    eval_probe = pd.concat(
        probes_to_combine,
        keys=split_names,
        names=["split"],
        axis=0,
    ) # non-random currently
    
    train_path = os.path.join(training_probe_dir, run_name + "_train.csv")
    train_data.to_csv(train_path)
    test_path = os.path.join(training_probe_dir, run_name + "_test.csv")
    test_data.to_csv(test_path)
    save_path = os.path.join(training_probe_dir, run_name + "_eval.csv")
    eval_probe.to_csv(save_path)

    dataset = Dataset.from_pandas(train_data)
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

    if os.path.exists(training_config.output_dir) and not training_config.overwrite_output_dir:
        raise ValueError(f"Output directory '{training_config.output_dir}' already exists. Set `overwrite_output_dir=True` to overwrite it.")

    print("Preparing data...")
    dataset, eval_dataset = prepare_steerability_probe(
        probe_config,
        model_config.model_name_or_path,
        training_config.run_name,
    )
    test_server(server_config, dataset[probe_config.source_text_col][0])

    print("Preparing trainer...")
    reward_funcs, reward_weights = get_reward_funcs(reward_config, model_config.model_name_or_path)
    training_config.reward_weights = reward_weights

    print("Reward functions:")
    print(*[f"{rf.__name__} (weight: {wt})" for rf, wt in zip(reward_funcs, reward_weights)], sep="\n")
    print()
    if reward_config.steering_goals is not None:
        print("Steering goals:")
        print(*reward_config.steering_goals, sep="\n")
        print()

    peft_config = get_peft_config(model_config)
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
        reward_extra_kwargs={
            "steering_goals": reward_config.steering_goals,
            "normalize_ortho": reward_config.normalize_ortho,
            "normalize_miscal": reward_config.normalize_miscal,
            "good_enough_threshold": reward_config.good_enough_threshold,
            "square_rewards": reward_config.square_rewards,
            "decile": reward_config.decile,
        },
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