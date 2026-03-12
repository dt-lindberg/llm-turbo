"""Run Qwen3.5-4B inference using vLLM for maximum throughput."""

import json
import os
import sys
import time

from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from logger import get_logger
from prompt import SYSTEM, USER

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", None)
TEMPERATURE = 0.6  # Recommended @ https://unsloth.ai/docs/models/qwen3.5
MAX_TOKENS = 2000
MODEL = "Qwen/Qwen3.5-4B"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: inference.py <name>", file=sys.stderr)
        sys.exit(1)
    name = sys.argv[1]
    log = get_logger(name)

    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN not set")

    import torch

    log.info(f"Torch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        log.info(f"GPU: {props.name}  VRAM: {props.total_memory / 1024**3:.2f} GB")
    else:
        raise RuntimeError("CUDA not available — model will fall back to CPU")

    # Load model with vLLM (PagedAttention + CUDA graphs)
    t_load_start = time.perf_counter()
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
    )
    t_load = time.perf_counter() - t_load_start
    log.info(f"Model loaded in {t_load:.2f}s")

    # Apply chat template
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=0.95,
        top_k=20,
        max_tokens=MAX_TOKENS,
    )

    log.info("Generating...")
    t_gen_start = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params)
    t_gen = time.perf_counter() - t_gen_start

    output = outputs[0].outputs[0]
    n_tokens = len(output.token_ids)
    response_text = output.text

    print(response_text, flush=True)

    log.info(
        f"Generated {n_tokens} tokens in {t_gen:.2f}s ({n_tokens / t_gen:.2f} tok/s)"
    )
    log.info(f"Model load time: {t_load:.2f}s")

    result = {
        "total_tokens": n_tokens,
        "generation_time": t_gen,
        "tokens_per_second": n_tokens / t_gen,
    }
    with open(f"outputs/{name}.json", "w") as f:
        json.dump(result, f, indent=2)
