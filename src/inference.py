"""Run Qwen3-4B inference locally using Unsloth FastLanguageModel."""

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
TEMPERATURE = 0.6
MAX_SEQ_LENGTH = 2048
MAX_TOKENS = 2000


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

    # Load pure text model — no vision encoder overhead
    t_load_start = time.perf_counter()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3-4B",
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
    inputs = tokenizer(
        text=input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    log.info("Generating...")
    t_gen_start = time.perf_counter()
    generated = model.generate(
        **inputs,
        max_new_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.95,
        top_k=20,
        use_cache=True,
    )
    t_gen = time.perf_counter() - t_gen_start

    new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
    n_tokens = len(new_tokens)
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

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
