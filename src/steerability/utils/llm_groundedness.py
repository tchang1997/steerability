from argparse import ArgumentParser
import json
import os
import re
import requests
from string import Template as StringTemplate

from beartype import beartype
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer

from steerability.utils.pairwise_goal_validation import (
    initialize_chat_instance,
    judge,
    maybe_truncate,
    VLLM_API_CONFIG,
)
from steerability.utils.result_utils import extract_oracle_best, STEERING_GOALS

from typing import Optional, Union

@beartype
def create_problem_set(
    probe: pd.DataFrame,
    prompt_template: Union[str, StringTemplate],
    source_col: str,
    response_col: str,
    seed: Optional[int] = 42,
    port: Optional[int] = 16384,
) -> pd.DataFrame:
    #probe_subset = probe.filter(regex=f"^({source_col}$|{response_col}$|{source_goal_prefix}|{response_goal_prefix})", axis=1)

    np.random.seed(seed)
    mask = np.random.rand(len(probe)) < 0.5
    probe["version_a"] = np.where(mask, probe[source_col], probe[response_col])
    probe["version_b"] = np.where(mask, probe[response_col], probe[source_col])
    probe["source_text_is_a"] = mask

    url = f"http://localhost:{port}/v1/models/"
    resp = requests.get(url)
    resp.raise_for_status()
    llm_name = resp.json()["data"][0]["id"]
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    probe["prompt"] = probe.apply(
        lambda row: prompt_template.substitute(
            version_a=maybe_truncate(row["version_a"], tokenizer),
            version_b=maybe_truncate(row["version_b"], tokenizer)
        ),
        axis=1,
    )
    return probe

def load_json_with_fallback(text: str):
    try:
        return json.loads(text)
    except Exception:
        print("JSON unloading exception raised on text. Attempting manual unload.")
        answer_match = re.search(r'"answer"\s*:\s*(\w+)', text)
        rationale_match = re.search(r'"rationale"\s*:\s*"([^"]+)"', text)

        parsed = {"answer": None, "rationale": None}
        if answer_match:
            parsed['answer'] = answer_match.group(1)
        if rationale_match:
            parsed['rationale'] = rationale_match.group(1)
        return parsed


@beartype
def parse_and_score(pset_with_answers: pd.DataFrame) -> pd.DataFrame:
    answers = pset_with_answers["clean_judge_output"].apply(load_json_with_fallback)
    final = answers.apply(pd.Series)
    return pd.concat([pset_with_answers, final], axis=1)

@beartype
def report_on_nos(nos: pd.DataFrame):
    console = Console()
    col_styles = {
        "text": "green",
        "llm_response": "yellow",
        "rationale": "bold cyan",
    }

    table = Table(title=f"`No` answers by judge (N={len(nos)})", show_header=True, header_style="bold magenta")
    for col, style in col_styles.items():
        table.add_column(col, style=style)
    for _, row in nos.iterrows():
        values = [str(row[c]) for c in col_styles.keys()]
        table.add_row(*values)
    console.print(table)


CACHE_PATH = "./cache/interactive_review_cache.json"
@beartype
def interactive_review(
    df: pd.DataFrame,
    spot_check_size: Optional[int] = 10,
    spot_check_seed: Optional[int] = 42,
    no_cache: Optional[bool] = False,
):
    # if not no_cache:
    #     if os.path.isfile(CACHE_PATH):
    #         with open(CACHE_PATH, "r") as f:
    #             cache = json.load(f)
    #     else:
    #         cache = {}
    # else:
    #     cache = None


    # Filter rows
    no_rows = df[df['answer'] != 'Yes'] # TODO: re-run and check
    yes_rows = df[df['answer'] == 'Yes'].sample(
        n=min(spot_check_size, len(df[df['answer'] == 'Yes'])),
        random_state=spot_check_seed
    )
    to_review = pd.concat([no_rows, yes_rows]).reset_index(drop=True)

    # Report
    print(f"\n{len(to_review)} rows to review (No: {len(no_rows)}, Yes: {len(yes_rows)}):")
    results = []
    console = Console()
    for idx, row in to_review.iterrows():
        table = Table(show_header=True, title=f"Example {idx + 1}/{len(to_review)}", header_style="white")
        table.add_column("Original", overflow="fold")
        table.add_column("Rewrite", overflow="fold")
        table.add_column("Answer", style="bold magenta")
        table.add_column("Reasoning", overflow="fold", style="bold cyan")
        table.add_row(row["text"], str(row["llm_response"]), row["answer"], row["rationale"])
        console.print(table)
        while True:
            user_input = input(">>> Approve reasoning? (yes/no): ").strip().capitalize()
            if user_input in ["Yes", "Y"]:
                approval = True
            elif user_input in ["No", "N"]:
                approval = False
            else: 
                print("Invalid input. Please type 'Yes' or 'No'.")
                continue

            results.append({
                'text': row['text'],
                'llm_response': row['llm_response'],
                'rationale': row['rationale'],
                'rationale_approved': approval,
                'spot_check': True
            })
            break


    df['rationale_approved'] = True
    df['spot_check'] = False

    # Update reviewed rows
    for res in results:
        mask = (
            (df['text'] == res['text']) &
            (df['llm_response'] == res['llm_response']) &
            (df['rationale'] == res['rationale'])
        )
        df.loc[mask, 'rationale_approved'] = res['rationale_approved']
        df.loc[mask, 'spot_check'] = True

    return df

if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--probe", type=str, required=True)
    psr.add_argument("--seed", type=int, default=42)
    psr.add_argument("--source-col", type=str, default="text")
    psr.add_argument("--response-col", type=str, default="llm_response")
    psr.add_argument("--prompt-file", type=str, default="./config/groundedness.prompt")
    psr.add_argument("--source-goal-prefix", type=str, default="source_")
    psr.add_argument("--response-goal-prefix", type=str, default="output_")
    psr.add_argument("--vllm-port", default=16384, type=int)
    psr.add_argument("--best-only", action="store_true")
    psr.add_argument("--spot-check-size", default=16, type=int)
    psr.add_argument("--name", type=str, required=True)
    args = psr.parse_args()

    path = f"judge_results/grounding/{args.name}_judged.csv"
    if os.path.exists(path):
        raise ValueError(f"{path} already exists. Pick a different name or move/delete the file.")

    with open(args.prompt_file, "r") as f:
        prompt_template = StringTemplate(f.read().strip())

    probe = pd.read_csv(args.probe, index_col=0)
    if args.best_only:
        orig_size = len(probe)
        print("Best-of-N file flag detected. Extracting best response for each goal.")
        probe = extract_oracle_best(probe, steering_goals=[g for g in STEERING_GOALS if g in probe.columns])
        print(f"After filtering: N = {orig_size} -> {len(probe)}")

    pset = create_problem_set(
        probe,
        prompt_template,
        args.source_col,
        args.response_col,
        args.seed,
        port=args.vllm_port,
    )
    chat_instance = initialize_chat_instance(
        args.vllm_port,
        api_config=VLLM_API_CONFIG,
        cache_suffix="_groundedness_judge.tsv"
    )
    pset_answered = judge(pset, chat_instance)
    df = parse_and_score(pset_answered)
    print("Evaluation results:")
    print(df["answer"].value_counts())

    df = interactive_review(df, args.spot_check_size)
    print("Saving results to", path)
    df.to_csv(path)