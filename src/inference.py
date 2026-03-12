"""Run Qwen3.5-4B inference locally using Unsloth."""

import json
import os
import sys
import time

from dotenv import load_dotenv
from unsloth import FastVisionModel

from prompt import SYSTEM, USER

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", None)
TEMPERATURE = 0.6  # Recommended @ https://unsloth.ai/docs/models/qwen3.5
MAX_SEQ_LENGTH = 16384


def format_message_for_vision(messages: list[dict]) -> list[dict]:
    """Convert plain-string message content to the vision-compatible format
    (list of typed dicts) that the Qwen3.5 processor expects."""
    formatted = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            formatted.append(
                {"role": msg["role"], "content": [{"type": "text", "text": content}]}
            )
        else:
            formatted.append(msg)
    return formatted


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: inference.py <name>", file=sys.stderr)
        sys.exit(1)
    name = sys.argv[1]

    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN not set")

    # Sanity-check environment
    import torch

    print(f"Torch version:  {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device:    {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM:           {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
    else:
        raise RuntimeError("CUDA not available — model will fall back to CPU")

    # Load model with Unsloth — 4-bit quantization for reduced VRAM usage
    # Qwen3.5 is always a vision model, so we use FastVisionModel
    t_load_start = time.perf_counter()
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name="unsloth/Qwen3.5-4B",
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        token=HF_TOKEN,
    )

    # Enable Unsloth's native 2x faster inference
    FastVisionModel.for_inference(model)
    t_load = time.perf_counter() - t_load_start
    print(f"Model loaded in {t_load:.2f}s", flush=True)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
    ]
    vision_messages = format_message_for_vision(messages)
    input_text = tokenizer.apply_chat_template(
        vision_messages, add_generation_prompt=True
    )
    inputs = tokenizer(
        text=input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    print("\nGenerating...", flush=True)
    t_gen_start = time.perf_counter()
    generated = model.generate(
        **inputs,
        max_new_tokens=8192,
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

    print("\n=== STATS ===\n")
    print(f"Model load time: {t_load:.2f}s")
    print(f"Generated {n_tokens} tokens in {t_gen:.2f}s ({n_tokens / t_gen:.2f} tok/s)")

    result = {
        "total_tokens": n_tokens,
        "generation_time": t_gen,
        "tokens_per_second": n_tokens / t_gen,
    }
    with open(f"outputs/{name}.json", "w") as f:
        json.dump(result, f, indent=2)
