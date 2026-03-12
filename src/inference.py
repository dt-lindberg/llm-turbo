"""Run Qwen3-0.6B with batched inference — amortize fixed per-step overhead."""

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
MAX_TOKENS = 2000
BATCH_SIZE = 32  # Load model weights once, generate BATCH_SIZE tokens per step


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

    # Duplicate prompt BATCH_SIZE times — same weights loaded, N tokens per step
    inputs = {k: v.expand(BATCH_SIZE, -1).contiguous() for k, v in single_inputs.items()}
    prompt_len = single_inputs["input_ids"].shape[1]

    log.info(f"Generating (batch_size={BATCH_SIZE})...")
    t_gen_start = time.perf_counter()

    with torch.inference_mode():
        # Prefill all batch items at once
        out = model(**inputs, use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        next_token = out.logits[:, -1:].argmax(dim=-1)  # [B, 1]
        new_token_ids = [next_token]

        # Pre-build position_ids for decode steps (same for all batch items)
        all_position_ids = (
            torch.arange(prompt_len, prompt_len + MAX_TOKENS, device="cuda")
            .view(1, -1)
            .expand(BATCH_SIZE, -1)
            .contiguous()
        )

        # Decode loop — BATCH_SIZE tokens per step, same overhead cost
        for step in range(MAX_TOKENS - 1):
            out = model(
                input_ids=next_token,
                past_key_values=past_kv,
                position_ids=all_position_ids[:, step : step + 1],
                use_cache=True,
                return_dict=True,
            )
            past_kv = out.past_key_values
            next_token = out.logits[:, -1:].argmax(dim=-1)  # [B, 1]
            new_token_ids.append(next_token)

    torch.cuda.synchronize()
    t_gen = time.perf_counter() - t_gen_start

    # Count all generated tokens across the full batch
    all_tokens = torch.cat(new_token_ids, dim=1)  # [B, MAX_TOKENS]
    n_tokens = int(all_tokens.numel())

    # Decode first sequence for display
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
