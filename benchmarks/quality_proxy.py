"""Download-free quality proxy: per-token perplexity on a fixed passage.

Compares, at equal bits/group:
  * bf16 reference
  * plain affine (MLX built-in, no rotation)
  * TurboQuant affine (rotation + affine)
  * TurboQuant LUT (rotation + non-uniform Lloyd-Max Metal kernel)

Isolates (a) the rotation benefit and (b) the non-uniform-codebook benefit.
"""

import argparse
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

import mlx_turboquant as tq
from mlx_turboquant.quant_linear import TurboQuantLinear, seed_for

TEXT = (
    "The history of quantization in signal processing traces back to Shannon's "
    "foundational work on source coding. Vector quantization seeks to represent "
    "high-dimensional data with a finite codebook while minimizing distortion. "
    "In modern language models, weights and key-value caches dominate memory, so "
    "low-bit quantization is essential for on-device inference. Random rotations "
    "spread outliers across coordinates, turning heavy-tailed weight distributions "
    "into approximately Gaussian ones that scalar quantizers handle gracefully. "
    "The optimal fixed-rate scalar quantizer for a Gaussian source is the "
    "Lloyd-Max quantizer, whose levels are the conditional means of each cell."
) * 3


def perplexity(model, tokens):
    ids = tokens[None, :]
    logits = model(ids[:, :-1])
    logits = logits.astype(mx.float32)
    targets = ids[:, 1:]
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tok_logp = mx.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    nll = -tok_logp.mean()
    mx.eval(nll)
    return math.exp(nll.item())


def plain_affine(model, bits, group_size):
    """Quantize a fresh copy with MLX affine, no rotation (rotation-off baseline)."""
    nn.quantize(model, group_size=group_size, bits=bits,
                class_predicate=lambda p, m: isinstance(m, nn.Linear) and "lm_head" not in p)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-0.6B-bf16")
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=64, dest="group_size")
    args = ap.parse_args()

    from mlx_lm import load

    print(f"model={args.model}  bits={args.bits}  group_size={args.group_size}\n")

    model, tok = load(args.model)
    tokens = mx.array(tok.encode(TEXT))

    ref = perplexity(model, tokens)
    print(f"  bf16 reference          ppl = {ref:8.3f}")

    # plain affine (rotation off)
    m2, _ = load(args.model)
    plain_affine(m2, args.bits, args.group_size)
    print(f"  plain affine (no rot)   ppl = {perplexity(m2, tokens):8.3f}")

    # turboquant affine (rotation on)
    m3, _ = load(args.model)
    from mlx_turboquant.quant_linear import turboquant_quantize
    turboquant_quantize(m3, bits=args.bits, group_size=args.group_size, mode="affine")
    print(f"  turboquant affine       ppl = {perplexity(m3, tokens):8.3f}")

    # turboquant LUT (rotation + non-uniform kernel)
    m4, _ = load(args.model)
    turboquant_quantize(m4, bits=args.bits, group_size=args.group_size, mode="lut")
    print(f"  turboquant LUT (kernel) ppl = {perplexity(m4, tokens):8.3f}")


if __name__ == "__main__":
    main()
