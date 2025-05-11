import os
import argparse
from peft import PeftModel
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

def merge_lora_and_save(base_model_path, lora_path, save_path):
    if os.path.exists(save_path):
        raise FileExistsError(f"Output path '{save_path}' already exists. Choose a new path or delete it first.")

    print("Saving config...")
    config = AutoConfig.from_pretrained(base_model_path)
    config.save_pretrained(save_path)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(save_path)

    # Load base model
    print("Loading base model:", base_model_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
    print("Base model dtype:", model.dtype)

    # Load LoRA adapter
    print("Loading LoRA adapter:", lora_path)
    model = PeftModel.from_pretrained(model, lora_path)

    # Merge LoRA into base weights
    print("Merging and unloading adapter...")
    model = model.merge_and_unload(progressbar=True)
    print("Final model dtype:", model.dtype)

    # Save merged model
    print("Saving...")
    model.save_pretrained(save_path)
    print(f"Merged model saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Path to base model")
    parser.add_argument("--lora", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--out", type=str, required=True, help="Where to save the merged model")
    args = parser.parse_args()

    merge_lora_and_save(args.base_model, args.lora, args.out)