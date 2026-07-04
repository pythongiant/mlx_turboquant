"""Non-uniform scalar codebooks — the "optimal scalar quantizer" of TurboQuant.

After the random rotation, every coordinate of a group looks like a (scaled)
standard normal.  TurboQuant then applies the *optimal scalar quantizer* for
that marginal.  For an MSE objective the optimal fixed-rate scalar quantizer is
the **Lloyd–Max** quantizer, whose reconstruction levels are the conditional
means of the source over each quantization cell.

Because the target marginal is fixed (unit-variance Gaussian after rotation and
per-group scaling), the levels are *data-oblivious constants* — we compute them
once with a deterministic weighted Lloyd iteration and reuse them for every
model.  This is what makes TurboQuant calibration-free.

Per group we store only a single scale ``s`` (the group RMS); the reconstructed
weight is ``ŵ = s · level[idx]`` where ``idx`` indexes into the shared codebook.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import numpy as np

__all__ = [
    "lloyd_max_levels",
    "levels_mx",
    "quantize_group",
    "dequantize_indices",
    "group_scale",
    "pack_indices",
    "unpack_indices",
    "quantize_weight_lut",
    "dequantize_weight_lut",
]

# Fine deterministic grid approximating N(0, 1); used both to fit the levels
# (weighted Lloyd) and as the empirical source for MSE evaluation in tests.
_GRID_LIMIT = 8.0
_GRID_POINTS = 200_001


@lru_cache(maxsize=1)
def _gaussian_grid():
    xs = np.linspace(-_GRID_LIMIT, _GRID_LIMIT, _GRID_POINTS)
    w = np.exp(-0.5 * xs * xs)
    w /= w.sum()
    return xs, w


@lru_cache(maxsize=None)
def lloyd_max_levels(bits: int, iters: int = 200) -> tuple:
    """Deterministic Lloyd–Max reconstruction levels for a unit Gaussian.

    Returns ``2**bits`` levels (sorted ascending) as a plain tuple of floats so
    the result is hashable/cacheable and trivially serializable.
    """
    if bits < 1 or bits > 8:
        raise ValueError("bits must be in [1, 8]")
    xs, w = _gaussian_grid()
    k = 1 << bits
    # Initialize levels at weighted quantiles for a stable, symmetric start.
    cdf = np.cumsum(w)
    qs = (np.arange(k) + 0.5) / k
    levels = np.interp(qs, cdf, xs)
    for _ in range(iters):
        # Assign each grid point to nearest level, then move level to the
        # weighted (conditional) mean of its cell — 1-D weighted Lloyd.
        idx = np.abs(xs[:, None] - levels[None, :]).argmin(axis=1)
        new = levels.copy()
        for j in range(k):
            m = idx == j
            wj = w[m].sum()
            if wj > 0:
                new[j] = (xs[m] * w[m]).sum() / wj
        if np.allclose(new, levels, atol=1e-9):
            levels = new
            break
        levels = new
    levels = np.sort(levels)
    # The Gaussian source is symmetric, so the optimal levels are exactly
    # antisymmetric; enforce it to remove grid-discretization asymmetry.
    levels = 0.5 * (levels - levels[::-1])
    return tuple(float(v) for v in levels)


@lru_cache(maxsize=None)
def levels_mx(bits: int) -> mx.array:
    """Codebook levels as an ``mx.array`` (float32), cached per bit-width."""
    return mx.array(lloyd_max_levels(bits), dtype=mx.float32)


def _midpoints(levels: mx.array) -> mx.array:
    return (levels[1:] + levels[:-1]) * 0.5


def quantize_group(x: mx.array, bits: int, scale: mx.array) -> mx.array:
    """Quantize ``x`` (already grouped, last axis = group) to codebook indices.

    ``scale`` broadcasts over the group axis (typically the per-group RMS).
    Returns integer indices in ``[0, 2**bits)`` with the same shape as ``x``.
    """
    levels = levels_mx(bits)
    xs = x / scale
    # Nearest level == bucket by the level midpoints (levels are sorted).
    bounds = _midpoints(levels)
    idx = mx.sum((xs[..., None] > bounds).astype(mx.int32), axis=-1)
    return idx.astype(mx.uint32)


def dequantize_indices(idx: mx.array, bits: int, scale: mx.array) -> mx.array:
    """Inverse of :func:`quantize_group`: ``ŵ = scale · level[idx]``."""
    levels = levels_mx(bits)
    return levels[idx.astype(mx.uint32)] * scale


def group_scale(x: mx.array, group_size: int, eps: float = 1e-8) -> mx.array:
    """Per-group RMS scale over the last axis, shape ``(..., n/group_size, 1)``."""
    n = x.shape[-1]
    if n % group_size != 0:
        raise ValueError(f"last dim {n} not divisible by group_size {group_size}")
    lead = x.shape[:-1]
    xg = x.reshape(*lead, n // group_size, group_size)
    s = mx.sqrt(mx.mean(xg * xg, axis=-1, keepdims=True)) + eps
    return s


def optimal_group_scale(
    x: mx.array, bits: int, group_size: int, iters: int = 4, eps: float = 1e-8
) -> mx.array:
    """Per-group scale that minimizes reconstruction MSE against a fixed codebook.

    RMS scaling is not MSE-optimal for a *fixed* non-uniform codebook.  For a
    frozen assignment the optimal scale is ``<x, L> / <L, L>`` (least squares);
    alternating this refit with re-assignment (a 1-D Lloyd on the scale) is what
    lets the Lloyd–Max codebook actually beat affine at equal bits.
    """
    levels = levels_mx(bits)
    n = x.shape[-1]
    lead = x.shape[:-1]
    xg = x.reshape(*lead, n // group_size, group_size)
    s = mx.sqrt(mx.mean(xg * xg, axis=-1, keepdims=True)) + eps
    for _ in range(iters):
        idx = quantize_group(xg, bits, s)
        L = levels[idx]
        num = mx.sum(xg * L, axis=-1, keepdims=True)
        den = mx.sum(L * L, axis=-1, keepdims=True) + eps
        s = mx.abs(num / den) + eps
    return s


def pack_indices(idx: mx.array, bits: int) -> mx.array:
    """Pack codebook indices (values in ``[0, 2**bits)``) along the last axis.

    ``32 // bits`` indices are packed little-endian into each ``uint32`` word,
    the same convention the Metal LUT kernel unpacks.  Requires the last
    dimension to be divisible by ``32 // bits`` (true for LLM ``in_features``
    with ``bits in {2, 4, 8}``).
    """
    per_word = 32 // bits
    k = idx.shape[-1]
    if k % per_word != 0:
        raise ValueError(f"last dim {k} not divisible by {per_word} (bits={bits})")
    lead = idx.shape[:-1]
    g = idx.reshape(*lead, k // per_word, per_word).astype(mx.uint32)
    shifts = (mx.arange(per_word, dtype=mx.uint32) * bits)
    return mx.sum(g << shifts, axis=-1).astype(mx.uint32)


def unpack_indices(packed: mx.array, bits: int, k: int) -> mx.array:
    """Inverse of :func:`pack_indices` (reference / test helper)."""
    per_word = 32 // bits
    lead = packed.shape[:-1]
    shifts = (mx.arange(per_word, dtype=mx.uint32) * bits)
    mask = (1 << bits) - 1
    vals = (packed[..., None] >> shifts) & mask
    return vals.reshape(*lead, k).astype(mx.uint32)


def quantize_weight_lut(w: mx.array, bits: int, group_size: int):
    """Symmetric non-uniform (Lloyd–Max) quantization of a weight matrix.

    Returns ``(packed_indices, scales)`` where ``scales`` is one per group.  No
    per-group bias is needed: the Gaussian source is zero-mean and the codebook
    is antisymmetric, so a single scale per group suffices (this is what makes
    the packed representation smaller than affine at the same bits).
    """
    out_f, in_f = w.shape
    scales = optimal_group_scale(w, bits, group_size)  # (out_f, in_f/gs, 1)
    wg = w.reshape(out_f, in_f // group_size, group_size)
    idx = quantize_group(wg, bits, scales)  # (out_f, in_f/gs, gs)
    idx = idx.reshape(out_f, in_f)
    packed = pack_indices(idx, bits)
    return packed, scales.reshape(out_f, in_f // group_size)


def dequantize_weight_lut(packed, scales, bits, group_size, in_f):
    """Pure-MLX reference dequantization for :func:`quantize_weight_lut`."""
    out_f = packed.shape[0]
    idx = unpack_indices(packed, bits, in_f).reshape(out_f, in_f // group_size, group_size)
    w = dequantize_indices(idx, bits, scales[..., None])
    return w.reshape(out_f, in_f)
