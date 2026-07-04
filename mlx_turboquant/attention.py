"""Attention seam: rotate the query to match TurboQuant's rotated keys.

``mlx_lm`` models call ``scaled_dot_product_attention`` (imported from
``mlx_lm.models.base``) after ``cache.update_and_fetch``.  For a
:class:`TurboQuantKVCache` the stored keys live in the rotated frame, so the
query must be rotated by the same RHT before the score matmul.  We wrap the
original ``scaled_dot_product_attention`` (which already dispatches quantized
caches to ``quantized_scaled_dot_product_attention``) and only inject the query
rotation — everything else is unchanged.

Because each model module binds its *own* reference to the function at import
time, we replace the symbol in ``base`` and in every already-imported
``mlx_lm.models.*`` module.
"""

from __future__ import annotations

import sys

import mlx.core as mx
from mlx.utils import tree_map

__all__ = ["patch_attention", "turbo_qjl_sdpa"]


def _apply_mask(scores, mask):
    if mask is None:
        return scores
    if isinstance(mask, str):  # "causal"
        qL, kL = scores.shape[-2:]
        q_idx = mx.arange(kL - qL, kL)
        k_idx = mx.arange(kL)
        mask = q_idx[:, None] >= k_idx[None]
    if mask.dtype == mx.bool_:
        return mx.where(mask, scores, mx.finfo(scores.dtype).min)
    return scores + mask


def turbo_qjl_sdpa(cache, rq, q_keys, q_values, scale, mask):
    """Quantized attention with the unbiased QJL residual correction.

    ``rq`` is the already-R-rotated query. Score = ``<Rq, k̂> + <Rq, r>`` where
    the first term is the usual quantized matmul against the stored key and the
    second is the QJL estimate of the residual inner product (which removes the
    MSE quantizer's score bias). The residual term is computed directly on the
    packed 1-bit sketch by a custom Metal kernel (``qjl_sign_dot``). Mirrors
    mlx-lm's quantized SDPA, adding the correction before the softmax.
    """
    from . import qjl
    from .kernels.qjl_dot import qjl_sign_dot

    B, n_q_heads, L, D = rq.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads
    off = cache.offset
    W = D // 32

    # Everything below works in the grouped (B, n_kv, n_repeats, L, ·) frame.
    sq = (rq * scale).reshape(B, n_kv_heads, n_repeats, L, D)
    q_proj = qjl.project_query(rq, cache.qjl_seed).reshape(
        B, n_kv_heads, n_repeats, L, D
    )
    keys = tree_map(lambda x: mx.expand_dims(x, -3), q_keys)
    values = tree_map(lambda x: mx.expand_dims(x, -3), q_values)

    scores = mx.quantized_matmul(
        sq, *keys, transpose=True, group_size=cache.group_size, bits=cache.bits
    )  # (B, n_kv, n_rep, L, off)

    # Packed-sign QJL correction via the Metal kernel.
    G = B * n_kv_heads
    q_proj_g = q_proj.reshape(G, n_repeats * L, D)
    packed_g = cache.sketch[..., :off, :].reshape(G, off, W)
    rnorm_c_g = (cache.rnorm[..., :off, 0] * qjl.qjl_constant(D)).reshape(G, off)
    corr = qjl_sign_dot(q_proj_g, packed_g, rnorm_c_g).reshape(
        B, n_kv_heads, n_repeats, L, off
    )
    scores = scores + scale * corr

    scores = _apply_mask(scores, mask)
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(
        scores, *values, transpose=False, group_size=cache.group_size, bits=cache.bits
    )
    return out.reshape(B, n_q_heads, L, D)


def _make_wrapper(orig):
    from .kv_cache import TurboQuantKVCache

    def scaled_dot_product_attention(
        queries, keys, values, cache=None, scale=1.0, mask=None, sinks=None, **kw
    ):
        if isinstance(cache, TurboQuantKVCache):
            rq = cache.rotate_query(queries)
            if cache.qjl and cache.sketch is not None:
                return turbo_qjl_sdpa(cache, rq, keys, values, scale, mask)
            queries = rq
        return orig(
            queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks, **kw
        )

    scaled_dot_product_attention._turboquant_wrapped = True
    return scaled_dot_product_attention


def patch_attention() -> None:
    import mlx_lm.models.base as base

    if getattr(base.scaled_dot_product_attention, "_turboquant_wrapped", False):
        return  # already patched

    orig = base.scaled_dot_product_attention
    base._turboquant_orig_sdpa = orig
    wrapped = _make_wrapper(orig)
    base.scaled_dot_product_attention = wrapped

    # Repoint every model module that imported the original symbol by value.
    for name, mod in list(sys.modules.items()):
        if not name.startswith("mlx_lm.models."):
            continue
        if getattr(mod, "scaled_dot_product_attention", None) is orig:
            mod.scaled_dot_product_attention = wrapped
