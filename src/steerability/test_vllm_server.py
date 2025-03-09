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
    psr.add_argument("--temperature", type=float, default=0.)
    psr.add_argument("--top-p", type=float, default=0.9)
    psr.add_argument("--max-tokens", type=int, default=512)
    psr.add_argument("-n", type=int, default=1)
    psr.add_argument("--min-p", type=float, default=0)
    psr.add_argument("--frequency-penalty", type=float, default=0.)
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
                temperature=args.temperature, 
                top_p=args.top_p,
                max_completion_tokens=args.max_tokens,
                n=args.n,
                frequency_penalty=args.frequency_penalty,
                extra_body={"min_p": args.min_p},
            )
        elapsed = time.time() - start
    except openai.APIConnectionError:
        print("Raised a connection error -- check the endpoint.")
        sys.exit(1)

    choices = completion.choices
    if len(choices) == 1:
        raw_response = choices[0].message.content
        print("--- RESPONSE ---")
        print(raw_response)
    else:
        for i, choice in enumerate(choices):
            raw_response = choice.message.content
            print(f"--- RESPONSE #{i+1} ---")
            print(raw_response)
            print()


    output_toks = len(raw_response.split())
    print("--- STATS --- ")
    print("Input tokens (est.):", input_toks)
    print("Output tokens (est.):", output_toks)
    print("Total time:", f"{elapsed:.2f}s")
    print("tok/s:", f"{(input_toks + output_toks) / elapsed:.2f}s")



