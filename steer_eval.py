if __name__ == '__main__': 
    pass
    # Stage 1: Read in default config

    # Stage 2: Look for steerability probe. If it doesn't exist, fail. Future: generate from scratch.

    # Stage 3: Model loading
    # If model is HuggingFace model, load vLLM model. Otherwise, ping OpenAI to check model exists. 
    # Otherwise, fail. Wait until connection established. 
    # > pid-vllm.out pid-vllm.err

    # Stage 4 (simultaneous): When endpoint healthy, begin evaluating steerability. Kill vLLM server on end. 

    # Stage 4 (simultaneous): Spin up Uvicorn goal-space server and pipe output
    # > pid-goalspace.out pid-goalspace.err

    # Stage 5: Map results to goal-space and save.

    # Stage 6: If LLM-as-judge flag on, spin up judge model from config. Wait until connection established, then eval and save.

    # Stage 7: After results, compute metrics, report, save as JSON.
