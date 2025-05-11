from argparse import ArgumentParser
import pickle
import re
from string import Template as StringTemplate

import pandas as pd
from sammo.base import Template
from sammo.components import Output, GenerateText
from tqdm.auto import tqdm

from steerability.utils.pairwise_goal_validation import initialize_chat_instance, VLLM_API_CONFIG
from steerability.utils.model_output_cleaner import clean_model_output

tqdm.pandas()

def get_cot_block(text):
    pattern = r"## Edits(.*?)## Rewritten text"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        content = match.group(1).strip()
    else:
        content = None
    return content

def call_llm(chat_instance, prompts):
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
    return final_output, raw_output

if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--instruction-file", type=str, nargs="+", required=True)
    psr.add_argument("--vllm-port", default=16384, type=int)
    psr.add_argument("--prompt-file", type=str, default="./config/extract_cot_instructions.prompt")
    psr.add_argument("--name", type=str, required=True)
    psr.add_argument("--nrows", type=int)
    args = psr.parse_args()

    with open(args.prompt_file, "r") as f:
        prompt_template = StringTemplate(f.read().strip())

    cots = pd.concat([pd.read_csv(f, index_col=0, nrows=args.nrows) for f in args.instruction_file], ignore_index=True)
    inst_block_raw = cots["raw_response"].progress_apply(get_cot_block)
    cots["prompts"] = [prompt_template.substitute(edits=inst_block) for inst_block in inst_block_raw]
    chat_instance = initialize_chat_instance(
        args.vllm_port,
        api_config=VLLM_API_CONFIG,
        cache_suffix="_cot_extractor.tsv"
    )
    clean_results, raw_results = call_llm(chat_instance, cots["prompts"])

    pattern = r"\d+\s*\.\s*(.+)"
    cots["inst_lists_extracted"] = clean_results
    cots["edit_instruction"] = cots["inst_lists_extracted"].apply(lambda res: re.findall(pattern, res))
    index_cols = cots.filter(regex="^delta_", axis=1).columns.tolist()
    final_inst_df = cots[index_cols + ["edit_instruction"]].set_index(index_cols)["edit_instruction"]
    cot_dict = {
        tuple(f"{name}_{val}" for name, val in zip(final_inst_df.index.names, idx)): inst
        for idx, inst in final_inst_df.items()
    }

    path = f"./data/cot/{args.name}_extracted_cot.pkl"
    with open(path, "wb") as f:
        pickle.dump(cot_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved CoT to", path)

