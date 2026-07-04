"""Packed-sign inner-product Metal kernel for the QJL residual correction.

The QJL score correction is ``corr[m,n] = ||r_n|| · √(π/2d) · Σ_d qproj[m,d]·s[n,d]``
where ``s[n,d] = ±1`` are the sign bits of ``RHT₂(residual_n)`` stored packed at
1 bit/channel.  Instead of unpacking the sketch to a dense ``±1`` array and doing
a full extra matmul (memory- and bandwidth-heavy), this kernel reads the packed
``uint32`` words directly and accumulates ``±qproj`` per bit — one 32-lane SIMD
group reduces over the head dimension ``D`` for each output ``(group, m, n)``.

Layout (leading batch/head dims flattened to ``G`` groups)::

    qproj   : (G, M, D)         float   — RHT₂(Rq) for the M queries of a group
    packed  : (G, NK, D/32)     uint32  — sign sketch of the NK stored keys
    rnorm_c : (G, NK)           float   — ‖r‖ already scaled by √(π/2d)
    out     : (G, M, NK)        float   — the correction term (add scale·out to scores)
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

__all__ = ["qjl_sign_dot"]

_SOURCE = """
    uint elem = thread_position_in_grid.y;   // flattened (g, m, n)
    uint lane = thread_position_in_grid.x;   // 0..31 reduction lane
    if (elem >= (uint)(G * M * NK)) return;

    uint n  = elem % NK;
    uint gm = elem / NK;
    uint m  = gm % M;
    uint g  = gm / M;

    constexpr uint W = D / 32u;
    uint q_base = (g * M + m) * D;
    uint p_base = (g * NK + n) * W;

    float acc = 0.0f;
    for (uint d = lane; d < (uint)D; d += 32u) {
        uint word = packed[p_base + (d >> 5)];
        uint bit  = (word >> (d & 31u)) & 1u;
        float s   = bit ? 1.0f : -1.0f;
        acc += (float)qproj[q_base + d] * s;
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        out[elem] = (T)(acc * (float)rnorm_c[g * NK + n]);
    }
"""


@lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(
        name="turboquant_qjl_dot",
        input_names=["qproj", "packed", "rnorm_c"],
        output_names=["out"],
        source=_SOURCE,
    )


def qjl_sign_dot(qproj: mx.array, packed: mx.array, rnorm_c: mx.array) -> mx.array:
    """Correction term ``(G, M, NK)`` from packed sign sketches (see module doc)."""
    G, M, D = qproj.shape
    NK = packed.shape[1]
    dtype = qproj.dtype
    return _kernel()(
        inputs=[qproj, packed, rnorm_c.astype(dtype)],
        template=[("T", dtype), ("G", G), ("M", M), ("NK", NK), ("D", D)],
        grid=(32, G * M * NK, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(G, M, NK)],
        output_dtypes=[dtype],
    )[0]
