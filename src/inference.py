"""Qwen3-0.6B: single-sample prefill + KV-expand for large batches.

All prompts are identical, so we can prefill once for batch=1 and
expand the resulting KV cache to any batch size. This avoids the
large-batch prefill OOM while keeping large-batch decode throughput.
"""

import json
import os
import sys
import time

import torch
from dotenv import load_dotenv
from unsloth import FastLanguageModel

from logger import get_logger
from prompt import SYSTEM, USER

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", None)
MAX_SEQ_LENGTH = 2048
MAX_TOKENS = 100   # Short decode; VRAM = BATCH_SIZE*(prompt_len+100)*130KB
BATCH_SIZE = 1024  # 1024*(100+100)*130KB + 1.2GB = 27.8GB — safe


def expand_kv(kv, batch_size: int):
    """Expand a batch-1 KV cache to batch_size.
    Handles both tuple-of-tuples and DynamicCache."""
    if isinstance(kv, tuple):
        return tuple(
            (
                k.expand(batch_size, -1, -1, -1).contiguous(),
                v.expand(batch_size, -1, -1, -1).contiguous(),
            )
            for k, v in kv
        )
    # DynamicCache (transformers ≥ 4.36)
    kv.key_cache = [
        k.expand(batch_size, -1, -1, -1).contiguous() for k in kv.key_cache
    ]
    kv.value_cache = [
        v.expand(batch_size, -1, -1, -1).contiguous() for v in kv.value_cache
    ]
    return kv


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: inference.py <name>", file=sys.stderr)
        sys.exit(1)
    name = sys.argv[1]
    log = get_logger(name)

    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN not set")

    log.info(f"Torch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        log.info(f"GPU: {props.name}  VRAM: {props.total_memory / 1024**3:.2f} GB")
    else:
        raise RuntimeError("CUDA not available — model will fall back to CPU")

    t_load_start = time.perf_counter()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3-0.6B",
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        token=HF_TOKEN,
    )
    FastLanguageModel.for_inference(model)
    t_load = time.perf_counter() - t_load_start
    log.info(f"Model loaded in {t_load:.2f}s")

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    single_inputs = tokenizer(
        text=input_text, add_special_tokens=False, return_tensors="pt"
    ).to("cuda")
    prompt_len = single_inputs["input_ids"].shape[1]

    log.info(f"Generating (batch_size={BATCH_SIZE}, max_tokens={MAX_TOKENS})...")
    t_gen_start = time.perf_counter()

    with torch.inference_mode():
        # Prefill with batch=1 — avoids large-batch prefill OOM
        out = model(**single_inputs, use_cache=True, return_dict=True)
        first_token = out.logits[:, -1:].argmax(dim=-1)  # [1, 1]

        # Expand KV cache to BATCH_SIZE (all prompts identical)
        past_kv = expand_kv(out.past_key_values, BATCH_SIZE)
        next_token = first_token.expand(BATCH_SIZE, -1).contiguous()  # [B, 1]
        new_token_ids = [next_token]

        # Pre-build position_ids for all decode steps on GPU
        all_position_ids = (
            torch.arange(prompt_len, prompt_len + MAX_TOKENS, device="cuda")
            .view(1, -1)
            .expand(BATCH_SIZE, -1)
            .contiguous()
        )

        # Decode loop — BATCH_SIZE tokens per step
        for step in range(MAX_TOKENS - 1):
            out = model(
                input_ids=next_token,
                past_key_values=past_kv,
                position_ids=all_position_ids[:, step : step + 1],
                use_cache=True,
                return_dict=True,
            )
            past_kv = out.past_key_values
            next_token = out.logits[:, -1:].argmax(dim=-1)
            new_token_ids.append(next_token)

    torch.cuda.synchronize()
    t_gen = time.perf_counter() - t_gen_start

    all_tokens = torch.cat(new_token_ids, dim=1)  # [B, MAX_TOKENS]
    n_tokens = int(all_tokens.numel())

    response_text = tokenizer.decode(all_tokens[0].tolist(), skip_special_tokens=True)
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
