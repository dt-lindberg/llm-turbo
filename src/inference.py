"""Qwen3-30B-A3B speculative decoding: Qwen3-0.6B draft → Qwen3-30B-A3B target.

Draft model generates 5 tokens at ~1000 tok/s; target verifies 5 tokens
in prefill mode (~11ms). Expected net speedup: 150-200+ tok/s vs 91 baseline.
"""

import json
import os
import sys
import time

import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from logger import get_logger
from llama_cpp.llama_speculative import LlamaDraftModel
from prompt import SYSTEM, USER

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", None)
TARGET_REPO = "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF"
TARGET_FILE = "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
DRAFT_REPO = "unsloth/Qwen3-0.6B-GGUF"
NUM_DRAFT_TOKENS = 5
MAX_TOKENS = 2000
N_CTX = 4096


class IncrementalDraft(LlamaDraftModel):
    """Wraps a small Llama GGUF model as draft for speculative decoding.
    Uses incremental KV cache: only re-evals new/changed tokens."""

    def __init__(self, draft_llm, num_pred_tokens: int = 5):
        self._llm = draft_llm
        self.num_pred_tokens = num_pred_tokens
        self._cached_n = 0

    def __call__(self, input_ids: np.ndarray, **kwargs) -> np.ndarray:
        n = len(input_ids)
        if self._cached_n > 0 and n >= self._cached_n:
            # Extend: only eval the new tokens since last call
            new = input_ids[self._cached_n:].tolist()
            if new:
                self._llm.eval(new)
        else:
            # Reset: tokens were rejected, rebuild from current prefix
            self._llm.reset()
            self._llm.eval(input_ids.tolist())
        self._cached_n = n

        tokens = []
        for _ in range(self.num_pred_tokens):
            tok = self._llm.sample(temp=0)
            tokens.append(tok)
            self._llm.eval([tok])
            self._cached_n += 1
        return np.array(tokens, dtype=np.intc)


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

    from llama_cpp import Llama

    log.info("Loading draft model (Qwen3-0.6B)...")
    draft_path = hf_hub_download(repo_id=DRAFT_REPO, filename=None,
                                  token=HF_TOKEN) if False else None
    # Find Q4_K_M for draft
    from huggingface_hub import list_repo_files
    draft_files = sorted(list_repo_files(DRAFT_REPO, token=HF_TOKEN))
    draft_q4 = [f for f in draft_files if "Q4_K_M" in f and f.endswith(".gguf")
                and "-of-" not in f]
    draft_file = draft_q4[0] if draft_q4 else next(
        f for f in draft_files if f.endswith(".gguf") and "-of-" not in f)
    log.info(f"Draft file: {draft_file}")
    draft_path = hf_hub_download(repo_id=DRAFT_REPO, filename=draft_file, token=HF_TOKEN)

    draft_llm = Llama(model_path=draft_path, n_gpu_layers=-1, n_ctx=N_CTX, verbose=False)
    draft_model = IncrementalDraft(draft_llm, num_pred_tokens=NUM_DRAFT_TOKENS)

    log.info("Loading target model (Qwen3-30B-A3B)...")
    target_path = hf_hub_download(repo_id=TARGET_REPO, filename=TARGET_FILE, token=HF_TOKEN)

    t_load_start = time.perf_counter()
    llm = Llama(
        model_path=target_path,
        n_gpu_layers=-1,
        n_ctx=N_CTX,
        verbose=False,
        draft_model=draft_model,
        logits_all=True,  # draft_model sets _logits_all=True internally but scores buffer defaults to (n_batch,vocab); pass explicitly to size it (n_ctx,vocab)
    )
    t_load = time.perf_counter() - t_load_start
    log.info(f"Target model loaded in {t_load:.2f}s")

    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]

    log.info("Generating with speculative decoding...")
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
    log.info(f"Target model load time: {t_load:.2f}s")

    result = {
        "total_tokens": n_tokens,
        "generation_time": t_gen,
        "tokens_per_second": n_tokens / t_gen,
    }
    with open(f"outputs/{name}.json", "w") as f:
        json.dump(result, f, indent=2)
