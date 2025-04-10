from argparse import ArgumentParser
import copy
import os
import requests

from ruamel.yaml import YAML
yaml = YAML(typ="safe")

if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--train-data", type=str, default="./training_probes/steerability_2d_in_3d_128x8_4xlopo_train.csv")
    psr.add_argument("--test-data", type=str, default="./training_probes/steerability_2d_in_3d_128x8_evaluation_test.csv")
    psr.add_argument("--template-cfg", type=str, default="./config/experiments/llama3.1_8b_2d_in_3d_checkpoint_0_zeroshot.yml")
    psr.add_argument("--cache-prefix", type=str, default="vllm-meta-llama3.1-8b-lora")
    psr.add_argument("--vllm-port", type=int, default=16384)
    psr.add_argument("--config-base-dir", type=str, default="./config/experiments/lora")
    psr.add_argument("--api-config", type=str, default=os.path.join(os.path.expanduser('~'), "api/vllm_oai.config"))
    args = psr.parse_args()

    # open template cfg
    with open(args.template_cfg, "r") as f:
        cfg_template = yaml.load(f)

    # ping vLLM to get list of LoRA adapters -> model names
    resp = requests.get(f"http://localhost:{args.vllm_port}/v1/models/")
    resp.raise_for_status()
    adapter_map = resp.json()["data"]  

    train_script_lines = ["#!/bin/bash\n", "set -x"]
    test_script_lines = ["#!/bin/bash\n", "set -x"]

    for adapter in adapter_map:
        adapter_name = adapter["id"]
        if "checkpoint" not in adapter_name:
            continue # skip the base model
        model_base, ckpt_name = adapter_name.rsplit("_", maxsplit=1)  # e.g. mylora_checkpoint-050
        run_name = adapter["root"].split("/")[-2]

        for mode in ["train", "test"]:
            cfg = copy.deepcopy(cfg_template)  # shallow copy should be fine unless nested dicts need deep copy
            cfg["experiment_name"] =  run_name + "/" + ckpt_name + "_" + mode 
            cfg["llm_settings"]["llm_name"] = adapter_name
            cfg["cache_file"] = f"{args.cache_prefix}_{adapter_name}.tsv"
            cfg["probe"] = args.train_data if mode == "train" else args.test_data

            out_dir = os.path.join(args.config_base_dir, run_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{ckpt_name}_{mode}.yml")

            with open(out_path, "w") as f:
                yaml.dump(cfg, f)

            print(f"Wrote config: {out_path}")

            cmd = f"python create_steerability_report.py --config {out_path} --api-config {args.api_config}"
            if mode == "train":
                train_script_lines.append(cmd)
            else:
                test_script_lines.append(cmd)

    train_script = os.path.join(args.config_base_dir, run_name, "eval_train.sh")
    test_script = os.path.join(args.config_base_dir, run_name, "eval_test.sh")

    with open(train_script, "w") as f:
        f.write("\n".join(train_script_lines) + "\n")
    os.chmod(train_script, 0o755)

    with open(test_script, "w") as f:
        f.write("\n".join(test_script_lines) + "\n")
    os.chmod(test_script, 0o755)

    print(f"Wrote script: {train_script}")
    print(f"Wrote script: {test_script}")