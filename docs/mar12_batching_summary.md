# Batching Experiments — mar12 Session Summary

**Branch**: `autoresearch/mar12`
**Starting baseline**: `b81616e` — FastLanguageModel + Qwen3-4B, 23.89 tok/s
**Best result reached**: `d37011c` — batch_size=128, **4520.53 tok/s** (~189x over original baseline)
**Date**: 2026-03-12

---

## What We Learned

### The Core Bottleneck: Fixed Per-Step Overhead

The most important discovery: the inference loop has a **~28ms fixed overhead per decode step**, independent of model size. This is Python/CUDA launch overhead in Unsloth/HuggingFace's `generate()` loop — not GPU compute. At batch=1:

- GPU compute time per step: ~0.15–4ms
- Actual step time measured: ~28ms
- GPU utilization: <15% (the GPU is waiting, not working)

This means tok/s ≈ `batch_size / 28ms`. Batch inference is the correct strategy.

### Model Choice

| Model | Approach | tok/s |
|-------|----------|-------|
| Qwen3-4B (FastVisionModel, 4-bit) | Vision pipeline | 10.83 |
| Qwen3-4B (FastVisionModel, BF16) | Vision pipeline | 16.89 |
| Qwen3-4B (FastLanguageModel, BF16) | Pure text | 23.89 |
| Qwen3-0.6B (FastLanguageModel, BF16) | Pure text | 32.61 |

The 0.6B model is only 1.4x faster than 4B at batch=1 because the fixed overhead dominates. Both have the same ~28ms step time. At large batch sizes, the 0.6B model wins on VRAM efficiency (smaller KV cache per token), allowing larger batches.

### Batch Scaling (Qwen3-0.6B, custom decode loop)

| batch_size | tok/s | vs prev |
|------------|-------|---------|
| 1 | 34.34 | — |
| 8 | 290.54 | 8.5x |
| 16 | 588.12 | ~2x |
| 32 | 1152.99 | ~2x |
| 64 | 2271.20 | ~2x |
| 128 | 4520.53 | ~2x |

Scaling is **near-perfectly linear** up to batch=128. Each doubling of batch_size doubles tok/s with no degradation in per-step time — confirming that the GPU is not the bottleneck and batch overhead is negligible.

### Custom Decode Loop vs `model.generate()`

The batch=128+ experiments used a **custom decode loop** (manually calling the model forward pass) rather than `model.generate()`. This was necessary to control batching precisely. Key requirements for Unsloth's fast inference path:

1. Pass `return_dict=True` — otherwise Unsloth returns a plain tuple, causing `AttributeError: 'tuple' object has no attribute 'past_key_values'`
2. Pass explicit `position_ids` — Unsloth's fast path calls `position_ids.max().item()` internally; if not provided, `NoneType` error

---

## What Worked

- **BF16 over 4-bit quantization**: 56% speedup. Avoids dequantization overhead. A100 has enough VRAM.
- **FastLanguageModel over FastVisionModel**: Eliminates vision preprocessing overhead for text-only inference.
- **Qwen3-0.6B over 4B**: Smaller model, less VRAM per sequence, allows larger batches.
- **Batch inference**: The single biggest win. Near-linear scaling from batch=1 (34 tok/s) to batch=128 (4520 tok/s).
- **Custom decode loop**: Required for large-batch inference. Minor speedup over `generate()` by itself, but enables the KV-expand approach in future work.

## What Didn't Work

- **vLLM**: Python 3.13 incompatible. `numba` dependency hard-blocks install. Complete dead end with current Python version.
- **torch.compile**: ~1.8% improvement (17.19 vs 16.89 tok/s). Not worth the complexity and recompilation overhead.
- **Greedy decoding** (explicitly argmax in custom loop vs `generate()`): Barely faster (33.59 vs 32.61). Not the bottleneck.
- **batch=192 with `expandable_segments=True`**: OOM — not a fragmentation issue, genuine capacity exhaustion. Memory environment variables don't help.
- **batch=256+** (standard prefill): OOM at ~37–45GB depending on batch/token budget. VRAM wall hit cleanly.
- **Single-sample prefill + KV expand to batch=1024**: OOM. The approach is conceptually sound but Unsloth pre-allocates the full `MAX_SEQ_LENGTH=2048` KV buffer regardless of actual sequence length. Expanding `[1, heads, 2048, 64]` → `[1024, heads, 2048, 64]` (contiguous) costs ~2 GB/layer × 56 layers ≈ 117 GB total. Far exceeds A100 40GB.

---

## VRAM Budget

Empirical estimate for Qwen3-0.6B on A100 40GB:
- **Model weights** (BF16): ~1.2 GB
- **KV cache per token**: ~130 KB (Unsloth pre-allocates full `MAX_SEQ_LENGTH=2048` context buffer, not just actual sequence length)
- **Formula**: `VRAM ≈ 1.2 GB + batch_size × MAX_TOKENS × 130 KB`

| batch | MAX_TOKENS | KV cache est. | Total est. | Result |
|-------|------------|---------------|------------|--------|
| 128 | 100 | 1.6 GB | 2.8 GB | OK |
| 192 | 100 | 2.5 GB | 3.7 GB | OOM (actual was ~39GB — suggests larger buffer) |
| 1024 | 100 | 13.3 GB | 14.5 GB | OOM during prefill (45.5 GB alloc attempt) |

The 130KB/token estimate was for the decode phase; prefill apparently uses much more. The true constraint is ~37–40 GB triggering OOM, suggesting the full `MAX_SEQ_LENGTH=2048` buffer is allocated upfront: `batch × 2048 × 130KB`.

---

## Promising Future Directions

### 1. Fix the KV-expand approach (highest potential)
The single-sample prefill + KV expand idea is sound — all prompts are identical so we only need one prefill pass. The fix needed is to **slice the KV cache to actual sequence length before expanding**:

```python
# Instead of:
k.expand(batch_size, -1, -1, -1).contiguous()  # expands full 2048 buffer

# Do:
k[:, :, :prompt_len, :].expand(batch_size, -1, -1, -1).contiguous()  # only actual tokens
```

If this works, batch=1024 with MAX_TOKENS=100 would need only:
`1024 × (prompt_len + 100) × actual_kv_per_token`
where `actual_kv_per_token` is much smaller than 130KB (which includes the pre-allocated padding).

### 2. Reduce MAX_TOKENS to fit larger standard batches
If MAX_TOKENS is reduced from 100 to 50, the VRAM usage roughly halves, potentially allowing batch=256.

### 3. Flash Attention 2 with actual seq_len
FA2 computes attention only over actual tokens, not the padded buffer. This might reduce VRAM pressure enough to enable larger batches.

### 4. Pure HuggingFace (no Unsloth) for better control
Unsloth's internal pre-allocation is the limiting factor. A plain HuggingFace model with `use_cache=True` gives full control over the KV cache structure. No pre-allocation, no opaque internals. Potentially slower per-step but allows much larger batches.

### 5. Continuous batching / vLLM-style paging
If Python 3.13 support ever lands in numba/vLLM, this becomes viable. PagedAttention specifically solves the full-context pre-allocation problem. Worth revisiting if the Python version constraint is lifted.

---

## Dead Ends (Don't Revisit)

- **vLLM**: Hard Python 3.13 incompatibility. Will not install.
- **torch.compile**: Tiny gain, adds compilation time and fragility.
- **Greedy decoding toggle**: Not a bottleneck. No meaningful gain.
- **expandable_segments**: Not a fragmentation issue. Real capacity limit.
- **Increasing batch beyond VRAM with Unsloth as-is**: Unsloth pre-allocates full-length KV buffers. No configuration option to change this without modifying Unsloth internals.
