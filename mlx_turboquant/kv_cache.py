"""TurboQuant KV cache — the inner-product (attention-score) regime.

TurboQuant's KV-cache insight: attention needs ``<q, k>`` to survive
quantization, and a random rotation *preserves inner products*
(``<Rq, Rk> == <q, k>``) while turning each rotated key vector into a
near-Gaussian, outlier-free distribution that low-bit scalar quantization
handles gracefully.  So we:

* rotate every key vector by a fixed Randomized Hadamard Transform over
  ``head_dim`` and store the *rotated* key, quantized;
* rotate the query by the *same* RHT at attention time (see ``attention.py``),
  so ``<Rq, dequant(Rk)> ≈ <q, k>``;
* leave values un-rotated (they are contracted with the attention
  probabilities, not with the query, so no inner product needs preserving).

This is the robust, inner-product-preserving core of the method; the paper's
optional 1-bit QJL residual (for a provably *unbiased* estimator) is a further
refinement documented in the README.

Implemented by subclassing ``mlx_lm``'s ``QuantizedKVCache`` and overriding only
``update_and_fetch`` — the parent quantizes keys *and* values and provides all
mask/state/serialization logic, so this stays robust across mlx-lm versions.
"""

from __future__ import annotations

import mlx.core as mx

from mlx_lm.models.cache import QuantizedKVCache

from . import qjl as _qjl
from .rotation import rht, supported_hadamard_block

__all__ = ["TurboQuantKVCache", "make_prompt_cache"]

# Fixed rotation seed: queries and keys of the same cache must share it; the
# actual value is irrelevant as long as it is consistent.
_KV_ROT_SEED = 0xC0FFEE


class TurboQuantKVCache(QuantizedKVCache):
    """Rotated (inner-product-preserving) KV cache.

    ``qjl=True`` additionally stores a 1-bit QJL sketch of the per-key
    quantization residual (``+1 bit/channel``), enabling the *unbiased*
    inner-product estimator in ``attention.py`` (removes the systematic
    score bias of the MSE quantizer for high-similarity query/key pairs).
    """

    def __init__(
        self,
        group_size: int = 64,
        bits: int = 4,
        seed: int = _KV_ROT_SEED,
        qjl: bool = False,
        qjl_seed: int = _qjl.QJL_SEED,
    ):
        super().__init__(group_size=group_size, bits=bits)
        self.seed = seed
        self.qjl = qjl
        self.qjl_seed = qjl_seed
        self._kblock = None  # Hadamard block for head_dim, resolved on first use
        self._d = None       # head_dim
        self.rnorm = None     # (B, n_kv, alloc, 1) residual norms
        self.sketch = None    # (B, n_kv, alloc, d/32) packed sign bits

    def _block(self, head_dim: int) -> int:
        if self._kblock is None:
            self._kblock = supported_hadamard_block(head_dim)
        return self._kblock

    def rotate_key(self, keys: mx.array) -> mx.array:
        return rht(keys, self.seed, self._block(keys.shape[-1]))

    def rotate_query(self, queries: mx.array) -> mx.array:
        return rht(queries, self.seed, self._block(queries.shape[-1]))

    def _ensure_side_buffers(self, template_keys):
        # Grow rnorm/sketch to match the parent's allocated key buffer length.
        B, n_kv, alloc = template_keys.shape[:3]
        d = self._d
        if self.sketch is None:
            self.rnorm = mx.zeros((B, n_kv, alloc, 1), dtype=mx.float16)
            self.sketch = mx.zeros((B, n_kv, alloc, d // 32), dtype=mx.uint32)
        elif self.sketch.shape[2] < alloc:
            pad = alloc - self.sketch.shape[2]
            self.rnorm = mx.concatenate(
                [self.rnorm, mx.zeros((B, n_kv, pad, 1), dtype=self.rnorm.dtype)], axis=2
            )
            self.sketch = mx.concatenate(
                [self.sketch, mx.zeros((B, n_kv, pad, d // 32), dtype=self.sketch.dtype)],
                axis=2,
            )

    def update_and_fetch(self, keys, values):
        rk = self.rotate_key(keys)
        if not self.qjl:
            return super().update_and_fetch(rk, values)

        # QJL path: also sketch the quantization residual of the new keys.
        self._d = keys.shape[-1]
        prev = self.offset
        qk, qv = super().update_and_fetch(rk, values)  # advances self.offset
        new = slice(prev, self.offset)
        khat = mx.dequantize(
            qk[0][..., new, :], qk[1][..., new, :], qk[2][..., new, :],
            group_size=self.group_size, bits=self.bits,
        )
        r = rk - khat
        rnorm_new, sketch_new = _qjl.sketch_residual(r, self.qjl_seed)
        self._ensure_side_buffers(qk[0])
        self.rnorm[..., new, :] = rnorm_new.astype(self.rnorm.dtype)
        self.sketch[..., new, :] = sketch_new
        return qk, qv


def make_prompt_cache(
    model, kv_bits: int = 4, kv_group_size: int = 64, qjl: bool = False
):
    """Per-layer TurboQuant KV cache for ``mlx_lm.generate(prompt_cache=...)``.

    Ensures the attention seam is installed (``register()``) so queries get
    rotated to match the stored rotated keys. Set ``qjl=True`` for the unbiased
    1-bit-residual estimator (costs +1 bit/channel).
    """
    from .patch import register

    register()
    n_layers = len(model.layers)
    return [
        TurboQuantKVCache(group_size=kv_group_size, bits=kv_bits, qjl=qjl)
        for _ in range(n_layers)
    ]
