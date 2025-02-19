from argparse import ArgumentParser
import sys
import time

import openai
from openai import OpenAI

if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--port", type=int, default=5000)
    psr.add_argument("--api-key", type=str, default="token-abc123")
    psr.add_argument("--model-tag", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    psr.add_argument("--message", type=str, default="Describe your capabilities.")
    args = psr.parse_args()

    base_url = f"http://localhost:{args.port}/v1"
    client = OpenAI(base_url=base_url, api_key=args.api_key)
    print("--- API REQUEST TEST ---")
    print("Endpoint:", base_url)
    print("Message:", args.message)
    input_toks = len(args.message.split()) # hack

    start = time.time()
    try:
        completion = client.chat.completions.create(
                model=args.model_tag,
                messages=[
                    {
                        "role": "user",
                        "content": args.message,
                    }
                ],
                temperature=0, # enforce determinism (ish)? for testing
            )
        elapsed = time.time() - start
    except openai.APIConnectionError:
        print("Raised a connection error -- check the endpoint.")
        sys.exit(1)

    print("--- RESPONSE ---")
    raw_response = completion.choices[0].message.content
    print(raw_response)

    output_toks = len(raw_response.split())
    print("--- STATS --- ")
    print("Input tokens (est.):", input_toks)
    print("Output tokens (est.):", output_toks)
    print("Total time:", f"{elapsed:.2f}s")
    print("tok/s:", f"{(input_toks + output_toks) / elapsed:.2f}s")



