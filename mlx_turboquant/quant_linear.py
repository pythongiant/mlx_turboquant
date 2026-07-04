"""``TurboQuantLinear`` — a rotated, quantized drop-in for ``nn.Linear``.

Forward pass (weight-only quantization)::

    y = quantized_matmul( rht(x), Wq )        # + bias

where ``Wq`` is the quantization of the *rotated* weight ``rht(W_rows)``.  Because
the RHT is orthogonal and the *same* rotation is applied to both the weight rows
(offline, in :meth:`from_linear`) and the activation (at runtime), the rotation
cancels out algebraically and only its variance-flattening effect on the
quantization survives::

    rht(W_row) · rht(x) == W_row · x        (see rotation.py / tests)

Phase 1 (this file) uses MLX's built-in **affine** ``quantized_matmul`` as the
scalar quantizer applied to the rotated weights — rotation alone removes the
outliers that dominate low-bit error, so this is already a strong, fully
runnable baseline.  Phase 2 swaps the affine path for a non-uniform Lloyd–Max
LUT matmul Metal kernel (``kernels/qmm.py``) to reach the paper's distortion.
"""

from __future__ import annotations

import zlib

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .rotation import rht, rotate_rows, supported_hadamard_block

__all__ = ["TurboQuantLinear", "turboquant_quantize", "seed_for"]


def seed_for(name: str, base_seed: int) -> int:
    """Deterministic, process-stable rotation seed for a layer path.

    Derived from the layer name so convert-time and load-time agree without
    storing a per-layer seed; ``base_seed`` lets the whole rotation be varied
    reproducibly and is recorded in the model config.
    """
    return (zlib.crc32(name.encode("utf-8")) ^ (base_seed & 0xFFFFFFFF)) & 0x7FFFFFFF


class TurboQuantLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int,
        group_size: int,
        seed: int,
        block: int,
        bias: bool = False,
        mode: str = "affine",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.seed = seed
        self.block = block
        self.mode = mode

        # Packed-parameter placeholders matching mx.quantize's output layout, so
        # that model.load_weights(...) can fill them from safetensors.
        self.weight = mx.zeros(
            (out_features, in_features * bits // 32), dtype=mx.uint32
        )
        n_groups = in_features // group_size
        self.scales = mx.zeros((out_features, n_groups), dtype=mx.float16)
        if mode == "affine":
            self.biases = mx.zeros((out_features, n_groups), dtype=mx.float16)
        if bias:
            self.bias = mx.zeros((out_features,), dtype=mx.float16)

    def _rotate(self, x: mx.array) -> mx.array:
        return rht(x, self.seed, self.block)

    def __call__(self, x: mx.array) -> mx.array:
        x = self._rotate(x)
        if self.mode == "lut":
            from .codebook import levels_mx
            from .kernels.qmm import turbo_qmm

            y = turbo_qmm(
                x,
                self["weight"],
                self["scales"],
                levels_mx(self.bits),
                self.bits,
                self.group_size,
                self.out_features,
                self.in_features,
            )
        else:
            biases = self["biases"] if "biases" in self else None
            y = mx.quantized_matmul(
                x,
                self["weight"],
                scales=self["scales"],
                biases=biases,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            )
        if "bias" in self:
            y = y + self["bias"]
        return y

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int,
        group_size: int,
        seed: int,
        mode: str = "affine",
    ) -> "TurboQuantLinear":
        out_features, in_features = linear.weight.shape
        block = supported_hadamard_block(in_features)
        has_bias = "bias" in linear

        m = cls(
            in_features,
            out_features,
            bits=bits,
            group_size=group_size,
            seed=seed,
            block=block,
            bias=has_bias,
            mode=mode,
        )
        w_rot = rotate_rows(linear.weight, seed, block)
        if mode == "lut":
            from .codebook import quantize_weight_lut

            packed, scales = quantize_weight_lut(w_rot, bits, group_size)
            m.weight = packed.astype(mx.uint32)
            m.scales = scales.astype(linear.weight.dtype)
        else:
            packed = mx.quantize(w_rot, group_size=group_size, bits=bits, mode=mode)
            m.weight = packed[0].astype(mx.uint32)
            m.scales = packed[1]
            if mode == "affine":
                m.biases = packed[2]
        if has_bias:
            m.bias = linear.bias
        return m


def _should_quantize(name: str, module: nn.Module, group_size: int, skip) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if any(s in name for s in skip):
        return False
    out_f, in_f = module.weight.shape
    # Built-in packing / grouping requirement for mx.quantized_matmul.
    if in_f % group_size != 0:
        return False
    return True


def turboquant_quantize(
    model: nn.Module,
    bits: int = 4,
    group_size: int = 64,
    base_seed: int = 0,
    mode: str = "affine",
    skip=("lm_head",),
    from_placeholders: bool = False,
):
    """Swap ``nn.Linear`` layers for :class:`TurboQuantLinear` in place.

    Mirrors ``mlx_lm.models.bitlinear_layers.bitnet_quantize``: walk the leaf
    modules, replace eligible ``nn.Linear`` layers, then ``update_modules``.

    * ``from_placeholders=False`` (convert time): quantize the real fp weights.
    * ``from_placeholders=True`` (load time): install empty
      :class:`TurboQuantLinear` shells with the right shapes so that
      ``model.load_weights`` can fill them from the safetensors.
    """
    new_layers = []
    for name, module in tree_flatten(
        model.leaf_modules(), is_leaf=nn.Module.is_module
    ):
        if not _should_quantize(name, module, group_size, skip):
            continue
        seed = seed_for(name, base_seed)
        if from_placeholders:
            out_f, in_f = module.weight.shape
            layer = TurboQuantLinear(
                in_f,
                out_f,
                bits=bits,
                group_size=group_size,
                seed=seed,
                block=supported_hadamard_block(in_f),
                bias=("bias" in module),
                mode=mode,
            )
        else:
            layer = TurboQuantLinear.from_linear(
                module, bits=bits, group_size=group_size, seed=seed, mode=mode
            )
        new_layers.append((name, layer))

    if new_layers:
        model.update_modules(tree_unflatten(new_layers))
    return model
