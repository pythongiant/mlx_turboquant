"""Randomized Hadamard Transform (RHT) — the "random rotation" of TurboQuant.

TurboQuant (https://arxiv.org/pdf/2504.19874) rotates each vector by a random
orthogonal matrix before scalar quantization.  A random rotation makes the
coordinates behave like an i.i.d. concentrated distribution (Beta / near
Gaussian), which (a) destroys the heavy outliers that wreck uniform quantizers
and (b) lets a single per-coordinate scalar quantizer be near MSE-optimal.

We realize the rotation as a *Randomized Hadamard Transform* (RHT), the same
data-oblivious rotation used by QuaRot / QuIP# / QJL::

    R = (1/sqrt(h)) . (I ⊗ H_h) . diag(d)

where ``H_h`` is a Hadamard matrix of a hardware-supported size ``h`` that
divides ``n``, ``I ⊗ H_h`` is a block-diagonal Hadamard (needed when ``n`` is not
itself a supported Hadamard size), and ``d`` is a deterministic ``±1`` vector
derived from an integer seed.  ``R`` is orthogonal, so ``R Rᵀ = I`` and inner
products are preserved (this is what the KV-cache path relies on).

Only the integer ``seed`` and block size are ever stored; ``d`` and the
transform are regenerated on the fly, so a rotation costs O(n log h) and no
matrix is materialized.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import numpy as np

__all__ = ["supported_hadamard_block", "signs_for", "rht", "rotate_rows"]


@lru_cache(maxsize=None)
def _hadamard_ok(h: int) -> bool:
    """Whether ``mx.hadamard_transform`` gives an *orthonormal* transform at ``h``.

    MLX accepts some sizes (e.g. 4608, 576) without error yet does not normalize
    them to an orthonormal transform under ``scale = 1/sqrt(h)`` (the norm is off
    by a constant factor).  Such sizes would break the ``R Rᵀ = I`` invariant, so
    we require actual norm preservation, not merely "does not throw".

    Uses a numpy probe and a CPU stream so it is safe to call from worker threads
    that have no default GPU stream (e.g. mlx-lm's threaded HTTP server).
    """
    if h < 1:
        return False
    try:
        x = mx.array(np.random.default_rng(0x5EED).standard_normal((1, h)).astype("float32"))
        with mx.stream(mx.cpu):
            y = mx.hadamard_transform(x, scale=1.0 / (h ** 0.5))
            ratio = (mx.sum(y * y) / mx.sum(x * x)) ** 0.5
            mx.eval(ratio)
        return bool(abs(ratio.item() - 1.0) < 1e-3)
    except Exception:
        return False


@lru_cache(maxsize=None)
def supported_hadamard_block(n: int) -> int:
    """Largest ``h`` dividing ``n`` for which ``mx.hadamard_transform`` works.

    For power-of-two / supported ``n`` this returns ``n`` (a full Hadamard).
    Otherwise it returns the largest supported divisor and the transform is
    applied block-diagonally (a Kronecker ``I ⊗ H_h``), which is still
    orthogonal and still whitens within each length-``h`` block.
    """
    if n < 1:
        raise ValueError(f"invalid rotation dimension {n}")
    # Prefer the largest divisor so we mix as many coordinates as possible.
    best = 1
    h = n
    # Walk divisors of n from largest to smallest.
    d = 1
    divisors = []
    while d * d <= n:
        if n % d == 0:
            divisors.append(d)
            divisors.append(n // d)
        d += 1
    for h in sorted(set(divisors), reverse=True):
        if _hadamard_ok(h):
            best = h
            break
    return best


@lru_cache(maxsize=None)
def signs_for(seed: int, n: int) -> mx.array:
    """Deterministic ``±1`` diagonal ``d`` of length ``n`` for a given seed.

    Generated with numpy's PCG64 (stable across platforms and MLX versions, and
    thread-safe — no GPU stream needed), so the same rotation is reproduced at
    convert time and at inference time, including inside worker threads.
    """
    rng = np.random.default_rng(seed)
    d = np.where(rng.random(n) < 0.5, -1.0, 1.0).astype("float32")
    return mx.array(d)


def _block_hadamard(x: mx.array, block: int) -> mx.array:
    """Apply a normalized Hadamard over contiguous blocks of the last axis."""
    n = x.shape[-1]
    if block == n:
        return mx.hadamard_transform(x, scale=1.0 / (block ** 0.5))
    lead = x.shape[:-1]
    xb = x.reshape(*lead, n // block, block)
    yb = mx.hadamard_transform(xb, scale=1.0 / (block ** 0.5))
    return yb.reshape(*lead, n)


def rht(x: mx.array, seed: int, block: int | None = None) -> mx.array:
    """Apply the RHT ``Rᵀ`` to the last axis of ``x``: ``d ⊙ blockH(x)``.

    Note this is ``Rᵀ`` (equivalently ``R`` up to the diagonal placement); it is
    its own consistent operator used for *both* activations and weight rows, so
    that ``(rht(W_row) · rht(x)) == (W_row · x)`` exactly.  ``block`` defaults to
    the largest supported Hadamard divisor of the last dimension.
    """
    n = x.shape[-1]
    if block is None:
        block = supported_hadamard_block(n)
    d = signs_for(seed, n).astype(x.dtype)
    return d * _block_hadamard(x, block)


def rotate_rows(w: mx.array, seed: int, block: int | None = None) -> mx.array:
    """Rotate the rows (last axis) of a weight matrix ``w`` by ``Rᵀ``.

    ``w`` has shape ``(out_features, in_features)``; the rotation is applied over
    ``in_features`` so it matches the activation rotation at inference time.
    """
    return rht(w, seed, block)
