import asyncio
from collections import defaultdict
import math
import re

from aiohttp import ClientSession, ServerDisconnectedError
import numpy as np

from utils.model_output_cleaner import clean_model_output, is_chat_completion_format

from beartype import beartype
from typing import List, Optional, Union

CHINESE_PATTERN = re.compile(r"[\u4E00-\u9FFF]")

async def send_request(
        session: ClientSession,
        texts: List[str],
        goals: Optional[List[str]] = None,
        port: Optional[int] = 12121,
        max_retries: Optional[int] = 10,
    ):
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
        if isinstance(kwargs["steering_goals"], list):
            if goal not in kwargs["steering_goals"]:
                continue
        sq_goal_err = np.square(np.array(kwargs[f"target_{goal}"]) - np.array(values))
        #print(f"MSE (goal: {goal}): {sq_goal_err.mean():.3f}")
        macro_negreward += sq_goal_err # add (\hat{z}_i - z*_i)^2 -> squared L2. Should throw a shape error if any goals are missing.
    rewards = (-np.sqrt(macro_negreward / len(mappings)) + 1) # normalize to [0, 1]; macro_negreward is in range [-sqrt(len(mappings), 0]
    return rewards

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