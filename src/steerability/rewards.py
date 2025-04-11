import asyncio
from collections import defaultdict
from functools import lru_cache
import math
import re

from aiohttp import ClientSession, ServerDisconnectedError
import numpy as np

from steerability.utils.model_output_cleaner import clean_model_output, is_chat_completion_format

from beartype import beartype
from typing import Any, List, Optional, Union

CHINESE_PATTERN = re.compile(r"[\u4E00-\u9FFF]")

async def send_request(
        session: ClientSession,
        texts: List[str],
        goals: Optional[List[str]] = None,
        port: Optional[int] = 12121,
        max_retries: Optional[int] = 10,
    ):
    """
        Yes, in theory, there are definitely easier ways than "rewards as a service," but if we want the rewards
        to implement any arbitrary function that might be hard to parallelize using standard Python, the server
        is nice and can, for example, host pre-trained models in paralell, call its own APIs -- anything without
        having to worry about pickle-ability
    """
    inference_url = f"http://127.0.0.1:{port}/goalspace" # why stop here? We can even throw vLLM in here someday
    payload = {"texts": texts}
    if goals is not None:
        payload["goals"] = goals
    retries = 0
    while retries < max_retries:
        try:
            async with session.post(inference_url, json=payload) as response:
                return await response.json()
        except ServerDisconnectedError:
            wait_time = 2 ** retries  # Exponential backoff
            print(f"Received a server disconnection error. Restart the goalspace-server ASAP. Retrying in {wait_time:.2f} seconds... ({retries+1}/{max_retries})")
            await asyncio.sleep(wait_time)
            retries += 1
        except Exception as e:
            print(f"Request failed due to {e}. Retrying...")
            await asyncio.sleep(1)
            retries += 1
    print("Max retries reached. Server did not restart in time.")

async def map_to_goalspace(
        texts: Union[str, List[str]],
        goals: Optional[List[str]] = None,
        port: Optional[int] = 12121,
        n_workers: Optional[int] = 16, # this is more how many chunks you want to split your workload into -- the request will just sit in a queue anyway
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

@lru_cache(maxsize=4096)
def get_mappings(completions: tuple[str, ...], model_name: str):
    completions = list(completions)
    cleaned_completions = [clean_model_output(model_name, completion) for completion in completions]
    mappings = asyncio.run(map_to_goalspace(cleaned_completions)) # this is cheap due to server-side caching
    return mappings

# can we cache somehow? overhead is probably small though since tuple() uses shallow copy
def prepare_completions_for_goalspace(completions: List[Any]) -> tuple[str]:
    if is_chat_completion_format(completions):
        completions = [completion[0]["content"] for completion in completions]
    completions_tuple = tuple(completions)
    return completions_tuple

def steerability_reward_wrapper(completions, **kwargs) -> Union[List[float], np.ndarray]: # TODO: ortho penalty, miscalibration, variance reg.?
    """
        In order to allow for *literally anything* to be used as a goal-space mapper, we delegate goalspace-mapping to an 
        external server and ping it with asyncio. This introduces some I/O overhead, yes, but `accelerate launch` + DeepSpeed ZeRO gets
        very confused and frustrated when I pass in a wrapper class that includes other model wrapper classes that cache inputs/outputs...
        goal-space mappings were not initially designed to be used directly as a reward model, so this is a happy medium (I think).
    """
    # this is z-hat
    completions_tuple = prepare_completions_for_goalspace(completions)
    mappings = get_mappings(completions_tuple, kwargs["model_name"][0])
    macro_negreward = np.zeros(len(completions))
    for goal, values in mappings.items():
        if isinstance(kwargs["steering_goals"], list):
            if goal not in kwargs["steering_goals"]:
                continue

        raw_clipped_err = np.clip(
                np.abs(np.array(kwargs[f"target_{goal}"]) - np.array(values)), 
                a_min=kwargs.get("good_enough_threshold", -float('inf')),
                a_max=kwargs.get("too_bad_threshold", float('inf')),
            ) # we currently don't use the too_bad threshold
        if kwargs.get("decile", False): # since granularity of 0.1 is specific enough to capture all differences
            raw_clipped_err = np.clip(np.ceil(raw_clipped_err * 10) / 10., 0.0, 1.0)
        sq_goal_err = np.square(raw_clipped_err)

        macro_negreward += sq_goal_err # add (\hat{z}_i - z*_i)^2 -> squared L2. Should throw a shape error if any goals are missing.
    if kwargs.get("rescale_norm", False):
        if kwargs.get("square_rewards", False):
            rewards =  1 - macro_negreward / len(mappings) # normalize to [0, 1]; macro_negreward is in range [len(mappings), 0]
        else:
            rewards = 1 - np.sqrt(macro_negreward / len(mappings))
    else:
        if kwargs.get("square_rewards", False):
            rewards = -macro_negreward # raw squared error
        else:
            rewards = -np.sqrt(macro_negreward)         
    return rewards

def scalar_rejection(a, b): # b onto a
    #return np.sqrt(np.linalg.norm(b, axis=1) ** 2 - (np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) + 1e-8)) ** 2)
    b_norm = np.sum(b * b, axis=1) + 1e-8
    proj = (np.sum(a * b, axis=1) / b_norm).reshape(-1, 1) * b  # shape [n, d]
    rejection_vec = a - proj
    rejection = np.linalg.norm(rejection_vec, axis=1)
    return rejection

def scalar_projection(a, b):
    return np.sum(a * b, axis=1) / (np.linalg.norm(b, axis=1) + 1e-8)

def orthogonality_wrapper(completions, **kwargs):
    completions_tuple = prepare_completions_for_goalspace(completions)
    mappings = get_mappings(completions_tuple, kwargs["model_name"][0])
    goals_to_evaluate = mappings.keys() if not isinstance(kwargs["steering_goals"], list) else kwargs["steering_goals"]
    z_star = np.array([kwargs[f"target_{goal}"] for goal in goals_to_evaluate]).T 
    z_0 = np.array([kwargs[f"source_{goal}"] for goal in goals_to_evaluate]).T 
    z_hat = np.array([mappings[goal] for goal in goals_to_evaluate]).T

    # (z_star - z_hat) -> oproj (z_hat - z_0)
    dist_from_source = z_hat - z_0  # shape [n, d]
    dist_requested = z_star - z_0  # shape [n, d]
    dist_to_goal = z_star - z_hat
    dist_from_goal_norm = np.linalg.norm(dist_from_source, axis=1) 
    dist_requested_norm = np.linalg.norm(dist_requested, axis=1)

    denom = dist_from_goal_norm + 1e-4 * (dist_requested_norm + 1e-8)
    rej = scalar_rejection(dist_to_goal, dist_requested)
    if kwargs.get("normalize_ortho", False):
        ortho = np.clip(rej / denom, 0, 1) 
    else:
        ortho = rej
    return -ortho 

def miscalibration_wrapper(completions, **kwargs):
    completions_tuple = prepare_completions_for_goalspace(completions)
    mappings = get_mappings(completions_tuple, kwargs["model_name"][0])
    goals_to_evaluate = mappings.keys() if not isinstance(kwargs["steering_goals"], list) else kwargs["steering_goals"]
    z_star = np.array([kwargs[f"target_{goal}"] for goal in goals_to_evaluate]).T 
    z_0 = np.array([kwargs[f"source_{goal}"] for goal in goals_to_evaluate]).T 
    z_hat = np.array([mappings[goal] for goal in goals_to_evaluate]).T
    
    dist_to_goal = z_star - z_hat
    dist_requested = z_star - z_0 
    denom = np.linalg.norm(dist_requested, axis=1) + 1e-8
    abs_proj_dist = np.abs(scalar_projection(dist_to_goal, dist_requested))
    if kwargs.get("normalize_miscal", False):
        miscal = abs_proj_dist / denom
    else:
        miscal = abs_proj_dist 
    return -miscal

def get_goal_neg_abs_err(completions, goal_name, **kwargs):
    completions_tuple = prepare_completions_for_goalspace(completions)
    mappings = get_mappings(completions_tuple, kwargs["model_name"][0]) 
    return -np.abs(np.array(kwargs[f"target_{goal_name}"]) - np.array(mappings[goal_name]))
    
def get_reading_difficulty_err(completions, **kwargs):
    return get_goal_neg_abs_err(completions, "reading_difficulty", **kwargs)

def get_text_length_err(completions, **kwargs):
    return get_goal_neg_abs_err(completions, "text_length", **kwargs)

def get_joy_err(completions, **kwargs):
    return get_goal_neg_abs_err(completions, "joy", **kwargs)

def get_surprise_err(completions, **kwargs):
    return get_goal_neg_abs_err(completions, "surprise", **kwargs)

def get_textual_div_error(completions, **kwargs):
    return get_goal_neg_abs_err(completions, "textual_diversity", **kwargs)

def get_formality_err(completions, **kwargs):
    return get_goal_neg_abs_err(completions, "formality", **kwargs)

def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    if is_chat_completion_format(completions):
        completions = [completion[0]["content"] for completion in completions]
    pattern = r"^<think>.*?</think>" # keep the deepseek format
    matches = [re.match(pattern, content, flags=re.DOTALL) for content in completions]
    return [1.0 if match else 0.0 for match in matches]

def english_only_reward_func(completions, **kwargs):
    """most common language mixing is Chinese, so just detect + penalize"""
    if is_chat_completion_format(completions):
        completions = [completion[0]["content"] for completion in completions]
    return [0.0 if CHINESE_PATTERN.search(content) else 1.0 for content in completions]

REGISTRY = {
    "steerability_error": steerability_reward_wrapper,
    "miscalibration": miscalibration_wrapper,
    "orthogonality": orthogonality_wrapper,
    "think_format": format_reward_func,
    "english_only": english_only_reward_func,
    "rd_error": get_reading_difficulty_err,
    "tl_error": get_text_length_err,
    "j_error": get_joy_err,
    "su_error": get_surprise_err,
    "td_error": get_textual_div_error,
    "fm_error": get_formality_err,
}