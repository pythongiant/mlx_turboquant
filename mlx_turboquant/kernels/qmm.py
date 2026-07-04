"""Non-uniform (Lloyd–Max) LUT dequant + matmul Metal kernel.

This is the kernel MLX cannot express with ``mx.quantized_matmul``: the weights
are indices into a *non-uniform* codebook (``levels``), not affine-uniform
codes.  The kernel computes, per output ``(m, n)``::

    y[m, n] = sum_k  x[m, k] * ( levels[idx[n, k]] * scale[n, k // group_size] )

directly on the packed indices — the dequantized weight matrix is never
materialized.  One 32-lane SIMD group cooperatively reduces over ``K`` for each
output element (the same structure as ``mlx_lm``'s ``bitlinear_matmul``).

Supported bit widths pack an integer number of indices per 32-bit word:
``bits in {2, 4, 8}`` (``32 // bits`` per word).  Other widths fall back to the
affine path in ``TurboQuantLinear``.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

__all__ = ["turbo_qmm", "supports_lut_bits"]

_LUT_BITS = (2, 4, 8)


def supports_lut_bits(bits: int) -> bool:
    return bits in _LUT_BITS


_SOURCE = """
    uint elem = thread_position_in_grid.y;      // flattened (m, n) output index
    uint lane = thread_position_in_grid.x;      // 0..31 reduction lane
    if (elem >= (uint)(M * N)) return;

    uint m = elem / N;
    uint n = elem % N;

    constexpr uint per_word = 32u / BITS;
    constexpr uint mask = (1u << BITS) - 1u;
    uint words_per_row = K / per_word;
    uint groups_per_row = K / GROUP_SIZE;

    float acc = 0.0f;
    for (uint k = lane; k < (uint)K; k += 32u) {
        uint word = packed_w[n * words_per_row + (k / per_word)];
        uint shift = (k % per_word) * BITS;
        uint idx = (word >> shift) & mask;
        float lvl = levels[idx];
        float s = (float)scales[n * groups_per_row + (k / GROUP_SIZE)];
        acc += (float)x[m * K + k] * lvl * s;
    }

    acc = simd_sum(acc);
    if (lane == 0) {
        out[m * N + n] = (T)acc;
    }
"""


@lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(
        name="turboquant_lut_qmm",
        input_names=["x", "packed_w", "scales", "levels"],
        output_names=["out"],
        source=_SOURCE,
    )


def turbo_qmm(
    x: mx.array,
    packed_w: mx.array,
    scales: mx.array,
    levels: mx.array,
    bits: int,
    group_size: int,
    out_features: int,
    in_features: int,
) -> mx.array:
    """Compute ``x @ dequant(packed_w).T`` with a non-uniform codebook.

    ``x`` has shape ``(..., in_features)``; the leading dims are flattened to a
    row count ``M``.  ``levels`` is the ``2**bits`` codebook (float32).
    """
    if not supports_lut_bits(bits):
        raise ValueError(f"LUT kernel supports bits in {_LUT_BITS}, got {bits}")

    orig_shape = x.shape
    if x.ndim > 2:
        x = x.reshape(-1, orig_shape[-1])
    elif x.ndim == 1:
        x = x[None, :]
    M, K = x.shape
    assert K == in_features, (K, in_features)
    N = out_features

    dtype = x.dtype
    levels = levels.astype(mx.float32)

    out = _kernel()(
        inputs=[x, packed_w, scales.astype(dtype), levels],
        template=[
            ("T", dtype),
            ("BITS", bits),
            ("GROUP_SIZE", group_size),
            ("M", M),
            ("N", N),
            ("K", K),
        ],
        grid=(32, M * N, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[dtype],
    )[0]

    if len(orig_shape) > 2:
        out = out.reshape(*orig_shape[:-1], N)
    elif len(orig_shape) == 1:
        out = out[0]
    return out
