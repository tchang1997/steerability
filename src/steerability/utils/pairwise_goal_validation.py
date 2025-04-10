from argparse import ArgumentParser
import json
import os
import re
import requests
from string import Template as StringTemplate

from beartype import beartype
import numpy as np
import pandas as pd
from rich.table import Table
from rich.console import Console
from ruamel.yaml import YAML
from sammo.base import Template
from sammo.components import Output, GenerateText
from sammo.throttler import AtMost
from scipy.stats import kendalltau
from transformers import AutoTokenizer

from steerability.custom_runners import VLLMOpenAIChat
from steerability.utils.model_output_cleaner import clean_model_output

from typing import Optional, Union

pd.options.mode.chained_assignment = None
yaml = YAML(typ='safe')

VLLM_API_CONFIG = os.path.join(os.path.expanduser("~"), "api/vllm_oai.config")
MAX_RESPONSE_LENGTH_FOR_JUDGE = 4096 # catch degenerate texts

def maybe_truncate(s: str, tokenizer: AutoTokenizer) -> str:
    input_ids = tokenizer.encode(s, truncation=True, max_length=MAX_RESPONSE_LENGTH_FOR_JUDGE)
    if len(input_ids) == MAX_RESPONSE_LENGTH_FOR_JUDGE:
        print(f"Truncated overlong response:", s[:5000] + "...")
        return tokenizer.decode(input_ids, skip_special_tokens=True)
    else:
        return s

@beartype
def create_problem_set(
    probe: pd.DataFrame,
    prompt_template: Union[str, StringTemplate],
    source_col: str,
    response_col: str,
    seed: Optional[int] = 42,
    source_goal_prefix: Optional[str] = "source_",
    response_goal_prefix: Optional[str] = "output_",
    tie_threshold: Optional[float] = 0.1,
    port: Optional[int] = 16384,
) -> pd.DataFrame:
    probe_subset = probe.filter(regex=f"^({source_col}$|{response_col}$|{source_goal_prefix}|{response_goal_prefix})", axis=1)

    np.random.seed(seed)
    mask = np.random.rand(len(probe_subset)) < 0.5
    probe_subset["version_a"] = np.where(mask, probe_subset[source_col], probe_subset[response_col])
    probe_subset["version_b"] = np.where(mask, probe_subset[response_col], probe_subset[source_col])
    attributes = [col[len(source_goal_prefix):] for col in probe_subset.columns if col.startswith(source_goal_prefix)]
    probe_subset["source_text_is_a"] = mask
    print(len(attributes), "attributes detected. Generating answer key.")
    for attr in attributes:
        source_attr = f"{source_goal_prefix}{attr}"
        response_attr = f"{response_goal_prefix}{attr}"
        diff = probe_subset[source_attr] - probe_subset[response_attr]
        tied = diff.abs() < abs(tie_threshold)
        source_greater = diff > abs(tie_threshold)
        source_less = diff < -abs(tie_threshold)
        probe_subset[f"answer_{attr}"] = np.select(
            [
                tied,
                (source_greater & mask) | (source_less & ~mask),
                (source_less & mask) | (source_greater & ~mask),
            ],
            [
                "Tie",
                "A",
                "B",
            ]
        )

    url = f"http://localhost:{port}/v1/models/"
    resp = requests.get(url)
    resp.raise_for_status()
    llm_name = resp.json()["data"][0]["id"]
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    probe_subset["prompt"] = probe_subset.apply(
        lambda row: prompt_template.substitute(
            version_a=maybe_truncate(row["version_a"], tokenizer),
            version_b=maybe_truncate(row["version_b"], tokenizer)
        ),
        axis=1,
    )
    return probe_subset

@beartype
def initialize_chat_instance(
    port: int,
    api_config: str,
    timeout: Optional[int] = 600,
    rate_limit: Optional[int] = 64,
    max_tokens: Optional[int] = 32000
):
    url = f"http://localhost:{port}/v1/models/"
    resp = requests.get(url)
    resp.raise_for_status()
    llm_name = resp.json()["data"][0]["id"]
    cache_file = llm_name.replace("/", "_") + "_as_judge.tsv"
    print(f"Detected vLLM instance at {url} running {llm_name}. Creating vLLM SAMMO runner with cache {cache_file}.")
    return VLLMOpenAIChat(
        model_id=llm_name,
        api_config=api_config, 
        cache=os.path.join("./cache", cache_file),
        timeout=timeout,
        rate_limit=AtMost(rate_limit, "running"),
        max_context_window=max_tokens,
        port=port,
    )

def judge(problem_set: pd.DataFrame, chat_instance: VLLMOpenAIChat) -> pd.DataFrame:
    prompts = problem_set["prompt"]
    outputs = Output(GenerateText(Template("{{input}}"), randomness=0.)).run(chat_instance, prompts.tolist())
    final_output = []
    raw_output = []
    for raw_resp in outputs.outputs.llm_responses: 
        try:
            iter_obj = raw_resp if isinstance(raw_resp[0], str) else raw_resp[0]
            for resp in iter_obj: # do we need [0]?
                clean_resp = clean_model_output(chat_instance._model_id, resp) # by default, only return one response
                raw_output.append(resp)
                final_output.append(clean_resp)
        except Exception as e:
            import traceback
            print("Exception raised during LLM response post-processing. This can happen if an LLM request failed for any reason. Rerun the current script to redo those calls. Successful calls will be fetched from the cache.")
            print("Full traceback:")
            print(traceback.format_exc())
            raise e
    problem_set["raw_judge_output"] = raw_output
    problem_set["clean_judge_output"] = final_output
    return problem_set

def flatten_judge_response(judge_json):
    flat = {}
    for k, v in judge_json.items():
        flat[f"{k}_choice"] = v.get("answer")
        flat[f"{k}_rationale"] = v.get("rationale")
    return pd.Series(flat)

def safe_json_load(res: str):
    try:
        return yaml.load(res)
    except Exception as e:
        print("Parse error -- attempting to fallback to regex-based output repair.\nOriginal message:", e)
        pattern = r'"(higher_[^"]+)":\s*{([^}]+)}'
        blocks = re.findall(pattern, res)
        
        result = {}
        for attr, inner_block in blocks:
            inner = {}
            for key_val in re.findall(r'"[^"]*"\s*:\s*"[^"]*"', inner_block):
                k, v = map(str.strip, key_val.split(":", 1))
                k = k.strip('"')
                v = v.strip('"')

                if v in {"A", "B", "Tie"}:
                    inner["answer"] = v
                else:
                    inner["rationale"] = v

            result[attr] = inner
        return result
    
def label_to_ordinal(series):
    return series.map({"A": 1, "Tie": 0, "B": -1})
    
def parse_and_score(pset_with_answers: pd.DataFrame) -> pd.DataFrame:
    judge_json = pset_with_answers["clean_judge_output"].apply(safe_json_load)
    score_df = judge_json.apply(flatten_judge_response)
    df = pd.concat([pset_with_answers, score_df], axis=1)

    results = {}
    attributes = [col[len("higher_"):-len("_choice")] for col in df.columns if col.startswith("higher_") and col.endswith("_choice")]
    for attr in attributes:
        try:
            judge_answers = label_to_ordinal(df[f"higher_{attr}_choice"])
            correct_answers = label_to_ordinal(df[f"answer_{attr}"])
        except KeyError:
            print("Attribute", attr, "not found. Skipping.")
            continue
        nan_mask = judge_answers.isna()
        if nan_mask.sum() > 0:
            print(f"Unrecoverable parse error resulted in {nan_mask.sum()} NaN(s) for attribute {attr}.")
        tau, p = kendalltau(judge_answers[~nan_mask], correct_answers[~nan_mask])
        #print(f"Kendall's tau ({attr}):", tau)
        results[attr] = {"tau": tau, "p": p, "n": (~nan_mask).sum()}
    return df, results

if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--probe", nargs="+", type=str, required=True)
    psr.add_argument("--pairs-to-sample", type=int, default=250)
    psr.add_argument("--seed", type=int, default=42)
    psr.add_argument("--source-col", type=str, default="text")
    psr.add_argument("--response-col", type=str, default="llm_response")
    psr.add_argument("--prompt-file", type=str, default="./config/llm_as_judge.prompt")
    psr.add_argument("--source-goal-prefix", type=str, default="source_")
    psr.add_argument("--response-goal-prefix", type=str, default="output_")
    psr.add_argument("--vllm-port", default=16384, type=int)
    psr.add_argument("--no-sample", action="store_true")
    psr.add_argument("--name", type=str)
    args = psr.parse_args()

    with open(args.prompt_file, "r") as f:
        prompt_template = StringTemplate(f.read().strip())

    if len(args.probe) == 1:
        probe = pd.read_csv(args.probe[0], index_col=0)
    elif len(args.probe) == 2:
        df1 = pd.read_csv(args.probe[0], index_col=0).add_prefix("left_")
        df2 = pd.read_csv(args.probe[1], index_col=0).add_prefix("right_")
        probe = pd.concat([
            df1, df2
        ], axis=1)
    else:
        raise ValueError("Can only compare original to rewrite (len(args.probe) == 1) or rewrite vs. rewrite (len(args.probe) == 2)")

    if not args.no_sample:
        probe = probe.sample(n=args.pairs_to_sample, replace=False, random_state=args.seed)
    pset = create_problem_set(
        probe,
        prompt_template,
        args.source_col,
        args.response_col,
        args.seed,
        source_goal_prefix=args.source_goal_prefix,
        response_goal_prefix=args.response_goal_prefix,
    )
    chat_instance = initialize_chat_instance(args.vllm_port, api_config=VLLM_API_CONFIG) # in the future, pass more args dynamically
    pset_answered = judge(pset, chat_instance)
    df, results = parse_and_score(pset_answered)

    table = Table(title="Kendall's Tau", show_lines=True)
    table.add_column("Attribute", justify="left", style="cyan", no_wrap=True)
    table.add_column("Tau", justify="right", style="magenta")
    table.add_column("P value", justify="right", style="magenta")

    for attr, stats in results.items():
        tau, p, n = stats["tau"], stats["p"], stats["n"]
        table.add_row(f"{attr} (N={n})", f"{tau:.4f}", f"{p:.4f}")

    # Print it
    console = Console()
    console.print(table)

    df.to_csv(f"judge_results/{args.name}_judged.csv")