from argparse import ArgumentParser
import json
import os
import pathlib

import pandas as pd

from sammo.base import Template
from sammo.components import GenerateText, Output
from sammo.data import DataTable
from sammo.extractors import LambdaExtractor
from sammo.runners import AzureChat 
from sammo.throttler import AtMost

PROMPT = "A group of three expert copy-editors were asked to rewrite some text following some instructions. Before answering, they were asked to explain their intended edits. Can you summarize everyone's proposed edits as a numbered list? Respond with only the numbered list and do not explain your answer."
AZURE_API_FILE = pathlib.Path().cwd().parent / "api" / "azure.openai" # "trapi.config"
AZURE_API_CONFIG = ""
if AZURE_API_FILE.exists():
    AZURE_API_CONFIG = AZURE_API_FILE

MODEL_NAME = "gpt-4-turbo"
CACHE_FILE = "llm_cot_extractor.tsv"
TIMEOUT = 120

if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--instruction-file", type=str, default="./data/llm_cot_outputs_raw.csv")
    psr.add_argument("--cols", nargs="+", type=str, default=["gpt4", "mixtral8x7b", "llama70b"])
    args = psr.parse_args()

    instructions = pd.read_csv(args.instruction_file, index_col=0)

    with open(AZURE_API_CONFIG, "r") as f:
        api_config = json.load(f)
    api_config["deployment_id"] = MODEL_NAME
    chat_instance = AzureChat( 
        model_id=MODEL_NAME,
        api_config=api_config,
        cache=os.path.join("./cache", CACHE_FILE),
        timeout=TIMEOUT,
        rate_limit=AtMost(3, "running"),
    )   
    
    dt = DataTable.from_pandas(instructions, output_fields=[], input_fields=["gpt4", "mixtral8x7b", "llama70b"], constants={"instructions": PROMPT})
    labeling_prompt = GenerateText(Template("{{constants.instructions}}\n\nStudent 1:{{input.gpt4}}\n\nStudent 2: {{input.mixtral8x7b}}\n\nStudent 3: {{input.llama70b}}"))
    final_out = Output(LambdaExtractor(labeling_prompt, 'lambda x: re.findall(r"^\d+\.\s*(.*)", x, re.MULTILINE)')).run(chat_instance, dt)
    list_of_instructions = final_out.outputs.values

    deltas = instructions[[c for c in instructions.columns if c.startswith("delta_")]].values
    goal_dict = {
        i: instr_list for i, instr_list in enumerate(list_of_instructions)
    }
    with open("data/grounded_instructions.json", "w") as f:
        json.dump(goal_dict, f)