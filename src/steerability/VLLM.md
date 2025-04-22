# Deepseek distilled
```
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve deepseek-ai/DeepSeek-R1 --dtype auto --port 5000 --api-key token-abc123 --host 127.0.0.1 --tensor-parallel-size 4 --m
ax-model-len 16384
```

