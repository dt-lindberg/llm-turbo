"""Qwen3-30B-A3B-Instruct-2507 via llama-cpp-python, full GPU offload (GGUF Q4_K_M)."""

import json
import os
import sys
import time

from dotenv import load_dotenv

from logger import get_logger
from prompt import SYSTEM, USER

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", None)

# Model is already downloaded at this location
MODEL_PATH = "/home/dlindberg/.cache/huggingface/hub/models--unsloth--Qwen3-30B-A3B-Instruct-2507-GGUF/snapshots/eea7b2be5805a5f151f8847ede8e5f9a9284bf77/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"

MAX_TOKENS = 2000
N_CTX = 4096
TEMPERATURE = 1.0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: inference.py <name>", file=sys.stderr)
        sys.exit(1)
    # name is the Slurm job_id
    name = sys.argv[1]
    log = get_logger(name)

    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN not set")

    from llama_cpp import Llama

    t_load_start = time.perf_counter()
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,  # all layers on GPU
        n_ctx=N_CTX,
        n_batch=2048,
        flash_attn=True,
        verbose=False,
    )
    t_load = time.perf_counter() - t_load_start
    log.info(f"Model loaded in {t_load:.2f}s")

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
    ]

    log.info("Generating...")
    t_gen_start = time.perf_counter()
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    t_gen = time.perf_counter() - t_gen_start

    n_tokens = response["usage"]["completion_tokens"]
    response_text = response["choices"][0]["message"]["content"]
    print(response_text, flush=True)

    log.info(
        f"Generated {n_tokens} tokens in {t_gen:.2f}s ({n_tokens / t_gen:.2f} tok/s)"
    )

    result = {
        "total_tokens": n_tokens,
        "generation_time": t_gen,
        "tokens_per_second": n_tokens / t_gen,
    }
    with open(f"outputs/{name}.json", "w") as f:
        json.dump(result, f, indent=2)
