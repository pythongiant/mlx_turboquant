"""1-bit QJL residual — the unbiased inner-product correction of TurboQuant.

The rotated-MSE key ``k̂ = dequant(quantize(Rk))`` gives a *biased* estimate of
the attention score: ``<Rq, k̂> = <Rq, Rk> - <Rq, r>`` where ``r = Rk - k̂`` is
the quantization residual, so it systematically *under*-counts by ``<Rq, r>``.

TurboQuant corrects this with a 1-bit **Quantized Johnson–Lindenstrauss (QJL)**
sketch of the residual.  Apply a second, independent Randomized Hadamard
Transform ``RHT₂`` (which Gaussianizes the residual's coordinates), keep only the
**sign** of each coordinate (1 bit/channel) plus the residual norm ``‖r‖``.  For
a random-direction JL map, ``E[ sign(<s,r>)·<s,q> ] = √(2/π)·<q,r>/‖r‖`` per
coordinate, so summing the ``d`` orthonormal RHT₂ coordinates gives the unbiased
estimator

    <Rq, r> ≈ √(π / 2d) · ‖r‖ · ⟨ RHT₂(Rq), sign(RHT₂(r)) ⟩ .

Total (unbiased) score: ``<Rq, k̂> + <Rq, r>`` — the first term exact from the
stored key, the second from the sketch.  Cost: +1 bit/channel and one extra
Hadamard on the query.  See https://arxiv.org/pdf/2504.19874 (and the QJL paper
arxiv 2406.03482 for the sign-JL estimator).
"""

from __future__ import annotations

import math

import mlx.core as mx

from .rotation import rht, supported_hadamard_block

__all__ = [
    "QJL_SEED",
    "qjl_constant",
    "pack_signs",
    "unpack_signs_pm1",
    "sketch_residual",
    "project_query",
]

# Second, independent rotation for the JL sketch (distinct from the KV rotation).
QJL_SEED = 0x51D3


def qjl_constant(d: int) -> float:
    """The estimator scale ``√(π / 2d)`` for a ``d``-dimensional RHT sketch."""
    return math.sqrt(math.pi / (2.0 * d))


def pack_signs(x: mx.array) -> mx.array:
    """Pack the sign bits of ``x`` (last axis, size divisible by 32) to uint32.

    Bit ``j`` is 1 iff ``x_j >= 0``.  Little-endian within each 32-bit word.
    """
    d = x.shape[-1]
    if d % 32 != 0:
        raise ValueError(f"sign-sketch dim {d} must be divisible by 32")
    lead = x.shape[:-1]
    bits = (x >= 0).astype(mx.uint32).reshape(*lead, d // 32, 32)
    shifts = mx.arange(32, dtype=mx.uint32)
    return mx.sum(bits << shifts, axis=-1).astype(mx.uint32)


def unpack_signs_pm1(packed: mx.array, d: int, dtype=mx.float32) -> mx.array:
    """Inverse of :func:`pack_signs`, returning ``±1`` values (1 -> +1, 0 -> -1)."""
    lead = packed.shape[:-1]
    shifts = mx.arange(32, dtype=mx.uint32)
    vals = (packed[..., None] >> shifts) & 1  # 0 / 1
    pm1 = vals.astype(dtype) * 2 - 1
    return pm1.reshape(*lead, d)


def sketch_residual(r: mx.array, seed: int = QJL_SEED):
    """Return ``(rnorm, packed_signs)`` for the residual ``r`` (last axis = d)."""
    d = r.shape[-1]
    block = supported_hadamard_block(d)
    y = rht(r, seed, block)  # RHT₂(r): Gaussianizes the residual coordinates
    rnorm = mx.sqrt(mx.sum(r * r, axis=-1, keepdims=True))
    return rnorm, pack_signs(y)


def project_query(q: mx.array, seed: int = QJL_SEED) -> mx.array:
    """Project a (rotated) query through the JL map: ``RHT₂(Rq)``."""
    d = q.shape[-1]
    return rht(q, seed, supported_hadamard_block(d))
