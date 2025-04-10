

#!/bin/bash

# Usage: CUDA_VISIBLE_DEVICES=5 bash load_lora_checkpoints_to_vllm.sh /path/to/lora_checkpoints/

set -e

BASE_DIR="$1"
if [[ -z "$BASE_DIR" ]]; then
  echo "Usage: bash $0 /path/to/lora_checkpoints/"
  exit 1
fi

BASE_DIR=$(realpath "$BASE_DIR")
BASE_NAME=$(basename "$BASE_DIR")

LORA_MODULES=""
for SUBDIR in "$BASE_DIR"/checkpoint-*; do
  if [[ -d "$SUBDIR" ]]; then
    NAME="${BASE_NAME}_$(basename "$SUBDIR")"
    LORA_MODULES+="${NAME}=${SUBDIR} "
  fi
done

echo "Loading LoRA modules:"
for entry in $LORA_MODULES; do
  echo "  * ${entry}"
done

# Launch vLLM
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
USE_FASTSAFETENSOR=true \
VLLM_USE_V1=1 \
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 16384 \
  --enable-lora \
  --dtype bfloat16 \
  --max-lora-rank 256 \
  --lora-dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.6 \
  --tensor-parallel-size 2 \
  --lora-modules $LORA_MODULES
