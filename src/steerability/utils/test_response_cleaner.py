from argparse import ArgumentParser
import glob
import json
import os

import pandas as pd

from steerability.instruction_generator import get_instruction_generator
from steerability.utils.model_output_cleaner import clean_model_output

RESULT_DIR = "./utils/response_cleaner_test_results/"

if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--cache-path", type=str, default="./cache/*.tsv")
    psr.add_argument("--csv", type=str)
    psr.add_argument("--instruction-generator", default="direct", type=str)
    psr.add_argument("--llm-name", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", type=str) # defaults to removing think tags. No-op for non think models.
    psr.add_argument("--raw-colname", default="raw", type=str)
    args = psr.parse_args()


    inst_gen = get_instruction_generator(args.instruction_generator)
    if args.csv is not None:
        print("CSV mode.")
        df = pd.read_csv(args.csv, index_col=0)
        llm_responses = df[args.raw_colname]
        model_cleaned = llm_responses.apply(lambda x: clean_model_output(args.llm_name, x))
        inst_cleaned = model_cleaned.apply(inst_gen.clean_response)
        combined = pd.concat([llm_responses, model_cleaned, inst_cleaned], keys=["raw", "-model", "-inst"], axis=1)
        mismatch_raw = combined[combined["raw"] != combined["-model"]]
        mismatch_inst = combined[combined["-model"] != combined["-inst"]]
        matches = combined[combined["raw"] == combined["-inst"]]
        print("Model-based cleaning modified", len(mismatch_raw), "responses")
        print("Instruction-based cleaning further modified", len(mismatch_inst), "responses")
        print()
        print("Raw->Model mismatch DF:")
        print(mismatch_raw[["raw", "-model"]].head())
        print()
        print("Model->Inst mismatch DF:")
        print(mismatch_inst[["-model", "-inst"]].head())
    else:
        print("TSV cache mode.")
        cache_files = glob.glob(args.cache_path)
        for file in cache_files:
            print("\nChecking cache:", file)
            raw_cache_tsv = pd.read_csv(file, sep="\t", header=None)
            llm_responses = raw_cache_tsv.loc[:, 1].apply(lambda x: json.loads(x)["choices"][0]["message"]["content"])
            model_cleaned = llm_responses.apply(lambda x: clean_model_output(args.llm_name, x))
            inst_cleaned = model_cleaned.apply(inst_gen.clean_response)
            combined = pd.concat([llm_responses, model_cleaned, inst_cleaned], keys=["raw", "-model", "-inst"], axis=1)
            mismatch_raw = combined[combined["raw"] != combined["-model"]]
            mismatch_inst = combined[combined["-model"] != combined["-inst"]]
            matches = combined[combined["raw"] == combined["-inst"]]
            print("Model-based cleaning modified", len(mismatch_raw), "responses")
            print("Instruction-based cleaning further modified", len(mismatch_inst), "responses")
            print()
            print("Raw->Model mismatch DF:")
            print(mismatch_raw[["raw", "-model"]].head())
            print()
            print("Model->Inst mismatch DF:")
            print(mismatch_inst[["-model", "-inst"]].head())

            basename = os.path.splitext(os.path.basename(file))[0]
            mismatch_raw.to_csv(os.path.join(RESULT_DIR, f"_raw_to_model_mismatch_{basename}.csv"))
            mismatch_inst.to_csv(os.path.join(RESULT_DIR, f"_raw_to_inst_mismatch_{basename}.csv"))
            matches.to_csv(os.path.join(RESULT_DIR, f"_all_match_{basename}.csv"))
