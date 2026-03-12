"""Qwen3-30B-A3B-Instruct-2507 via llama-cpp-python, full GPU offload (GGUF Q4_K_M)."""

import json
import os
import sys
import time

import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, list_repo_files

from logger import get_logger
from prompt import SYSTEM, USER

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", None)
REPO_ID = "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF"
MAX_TOKENS = 2000
N_CTX = 4096

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: inference.py <name>", file=sys.stderr)
        sys.exit(1)
    name = sys.argv[1]
    log = get_logger(name)

    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN not set")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        log.info(f"GPU: {props.name}  VRAM: {props.total_memory / 1024**3:.2f} GB")
    else:
        raise RuntimeError("CUDA not available")

    # Find the Q4_K_M GGUF file (single-part preferred, else first shard)
    all_files = sorted(list_repo_files(REPO_ID, token=HF_TOKEN))
    q4_files = [f for f in all_files if "Q4_K_M" in f and f.endswith(".gguf")]
    single = [f for f in q4_files if "-of-" not in f]
    filename = single[0] if single else q4_files[0]
    log.info(f"Downloading: {filename}")

    model_path = hf_hub_download(repo_id=REPO_ID, filename=filename, token=HF_TOKEN)
    log.info(f"Model path: {model_path}")

    from llama_cpp import Llama

    t_load_start = time.perf_counter()
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,  # all layers on GPU
        n_ctx=N_CTX,
        flash_attn=True,
        verbose=False,
    )
    t_load = time.perf_counter() - t_load_start
    log.info(f"Model loaded in {t_load:.2f}s")

    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]

    log.info("Generating...")
    t_gen_start = time.perf_counter()
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0,
    )
    t_gen = time.perf_counter() - t_gen_start

    n_tokens = response["usage"]["completion_tokens"]
    response_text = response["choices"][0]["message"]["content"]
    print(response_text, flush=True)

    log.info(f"Generated {n_tokens} tokens in {t_gen:.2f}s ({n_tokens / t_gen:.2f} tok/s)")
    log.info(f"Model load time: {t_load:.2f}s")

    result = {
        "total_tokens": n_tokens,
        "generation_time": t_gen,
        "tokens_per_second": n_tokens / t_gen,
    }
    with open(f"outputs/{name}.json", "w") as f:
        json.dump(result, f, indent=2)
