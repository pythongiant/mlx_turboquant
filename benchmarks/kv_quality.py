"""KV-cache quality: teacher-forced perplexity with a quantized KV cache.

A single forward over the sequence *with a cache* writes each position's K/V
into the (quantized) cache and attends over them causally, so the resulting
per-token NLL directly reflects KV-cache quantization error.  Compares plain
affine KV vs TurboQuant (rotated) KV at several bit widths.
"""

import argparse
import math

import mlx.core as mx

from mlx_lm.models.cache import KVCache, QuantizedKVCache

import mlx_turboquant as tq
from mlx_turboquant.kv_cache import TurboQuantKVCache

TEXT = (
    "The history of quantization in signal processing traces back to Shannon's "
    "foundational work on source coding. Vector quantization seeks to represent "
    "high-dimensional data with a finite codebook while minimizing distortion. "
    "In modern language models, weights and key-value caches dominate memory. "
    "Random rotations spread outliers across coordinates, turning heavy-tailed "
    "distributions into approximately Gaussian ones. Because an orthogonal "
    "rotation preserves inner products, attention scores computed on rotated "
    "keys and queries are unchanged, while the rotated keys quantize far more "
    "gracefully at low bit-widths. This is the core idea behind TurboQuant's "
    "inner-product-preserving key-value cache quantization scheme for "
    "long-context transformer inference on device."
) * 4


def ppl_with_cache(model, tokens, cache):
    ids = tokens[None, :]
    logits = model(ids[:, :-1], cache=cache).astype(mx.float32)
    targets = ids[:, 1:]
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tok_logp = mx.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    nll = -tok_logp.mean()
    mx.eval(nll)
    return math.exp(nll.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-0.6B-bf16")
    ap.add_argument("--group-size", type=int, default=64, dest="group_size")
    args = ap.parse_args()

    from mlx_lm import load

    tq.register()
    model, tok = load(args.model)
    tokens = mx.array(tok.encode(TEXT))
    n = len(model.layers)
    print(f"model={args.model}  seq_len={tokens.size}  group_size={args.group_size}\n")

    ref = ppl_with_cache(model, tokens, [KVCache() for _ in range(n)])
    print(f"  fp16 KV (reference)     ppl = {ref:8.3f}\n")
    print(f"  {'bits':>4}   {'affine KV':>12}   {'TurboQuant KV':>14}   "
          f"{'TQ+QJL (+1b/ch)':>16}")
    for bits in (8, 4, 3, 2):
        aff = ppl_with_cache(
            model, tokens,
            [QuantizedKVCache(group_size=args.group_size, bits=bits) for _ in range(n)],
        )
        tqv = ppl_with_cache(
            model, tokens,
            [TurboQuantKVCache(group_size=args.group_size, bits=bits) for _ in range(n)],
        )
        qjl = ppl_with_cache(
            model, tokens,
            [TurboQuantKVCache(group_size=args.group_size, bits=bits, qjl=True)
             for _ in range(n)],
        )
        print(f"  {bits:>4}   {aff:>12.3f}   {tqv:>14.3f}   {qjl:>16.3f}")


if __name__ == "__main__":
    main()
