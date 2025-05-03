#!/bin/bash

stage_id="$1"

# Define model and path mappings
declare -A model_names
declare -A model_paths

model_names["1a"]="meta-llama/Meta-Llama-3.1-8B-Instruct"
model_paths["1a"]="/data2/ctrenton/llm/steerability_tuning_v2_stage_1a_learn_fk_128x8"

model_names["1b"]="/data2/ctrenton/merged/_v2_stage_1a_learn_fk_64x8"
model_paths["1b"]="/data2/ctrenton/llm/steerability_tuning_v2_stage_1b_learn_fk_wo_hd_128x8"

model_names["1c"]="/data2/ctrenton/merged/_v2_stage_1b_learn_fk_wo_hd_64x8"
model_paths["1c"]="/data2/ctrenton/llm/steerability_tuning_v2_stage_1c_learn_hd_v2_128x8/"

model_names["1d"]="/data2/ctrenton/merged/_v2_stage_1c_learn_hd_v2_64x8"
model_paths["1d"]="/data2/ctrenton/llm/steerability_tuning_v2_stage_1d_hd_wo_fk_64x8"

model_names["2a"]="/data2/ctrenton/merged/_v2_stage_1d_learn_hd_wo_fk_64x8"
model_paths["2a"]="/data2/ctrenton/llm/steerability_tuning_v2_stage_2a_full_64x8"

model_names["2b"]="/data2/ctrenton/merged/_v2_stage_2a_steer_tuning"
model_paths["2b"]="/data2/ctrenton/llm/steerability_tuning_v2_stage_2b_full_normalized_64x8"

# Check if stage_id is valid
if [[ -z "${model_names[$stage_id]}" ]]; then
  echo "Unknown stage_id: $stage_id"
  exit 1
fi

lora_arg="stage-${stage_id}=${model_paths[$stage_id]}"
echo "Preview command:"
cat <<EOF
USE_FASTSAFETENSOR=true NCCL_P2P_DISABLE=1 \\
vllm serve "${model_names[$stage_id]}" \\
  --port 16384 \\
  --enable-lora \\
  --dtype bfloat16 \\
  --lora-dtype bfloat16 \\
  --max-model-len 32000 \\
  --gpu-memory-utilization 0.6 \\
  --tensor-parallel-size 2 \\
  --max-lora-rank 256 \\
  --lora-modules "$lora_arg"
EOF

USE_FASTSAFETENSOR=true NCCL_P2P_DISABLE=1 \
vllm serve "${model_names[$stage_id]}" \
  --port 16384 \
  --enable-lora \
  --dtype bfloat16 \
  --lora-dtype bfloat16 \
  --max-model-len 32000 \
  --gpu-memory-utilization 0.6 \
  --tensor-parallel-size 2 \
  --max-lora-rank 256 \
  --lora-modules "$lora_arg"