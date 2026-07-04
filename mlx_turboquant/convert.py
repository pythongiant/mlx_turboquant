"""Convert a full-precision model into a TurboQuant model directory.

The output is a standard mlx-lm model directory (safetensors + config.json +
tokenizer) whose ``config.json`` carries a ``quantization_config`` with
``quant_method == "turboquant"``.  After ``tq.register()`` it loads through the
ordinary ``mlx_lm`` entry points; without registration use ``tq.load``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mlx.core as mx
from mlx.utils import tree_map_with_path

from .quant_linear import turboquant_quantize

__all__ = ["convert"]

_DTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}


def _resolve_mode(mode: str, bits: int) -> str:
    """Pick the scalar quantizer applied to the (rotated) weights.

    'auto' -> 'affine'.  Rotation is the robust, always-on win for weight
    quantization; MLX's affine ``quantized_matmul`` is *data-adaptive* (per-group
    scale AND bias) and both fast and near-lossless at >=4 bits, so it is the
    sensible default there.  The non-uniform Lloyd-Max 'lut' Metal kernel wins on
    reconstruction MSE at very low bits and is available opt-in (`--mode lut`),
    but it is slower and does not beat adaptive affine end-to-end at 4 bits.
    """
    if mode != "auto":
        return mode
    return "affine"


def convert(
    hf_path: str,
    mlx_path: str = "mlx_tq_model",
    bits: int = 4,
    group_size: int = 64,
    rot_base_seed: int = 0,
    dtype: Optional[str] = None,
    skip=("lm_head",),
    mode: str = "auto",
    upload_repo: Optional[str] = None,
):
    from mlx_lm.utils import save

    mode = _resolve_mode(mode, bits)
    mlx_path = Path(mlx_path)
    if mlx_path.exists():
        raise ValueError(f"{mlx_path} already exists; pick a new --out path.")

    print(f"[turboquant] loading {hf_path}")
    from mlx_lm import load as mlx_load

    model, tokenizer, config = mlx_load(hf_path, return_config=True, lazy=True)

    if dtype is None:
        dtype = config.get("torch_dtype")
    if dtype in _DTYPES:
        tdtype = _DTYPES[dtype]
        cast_predicate = getattr(model, "cast_predicate", lambda _: True)

        def _set(k, v):
            if cast_predicate(k) and mx.issubdtype(v.dtype, mx.floating):
                return v.astype(tdtype)
            return v

        model.update(tree_map_with_path(_set, model.parameters()))

    print(
        f"[turboquant] rotating + quantizing to {bits}-bit "
        f"(group_size={group_size}, quantizer={mode})"
    )
    turboquant_quantize(
        model,
        bits=bits,
        group_size=group_size,
        base_seed=rot_base_seed,
        mode=mode,
        skip=tuple(skip),
        from_placeholders=False,
    )

    quantization_config = {
        "quant_method": "turboquant",
        "bits": bits,
        "group_size": group_size,
        "mode": mode,
        "rot_base_seed": rot_base_seed,
        "skip": list(skip),
        "rotation": "randomized_hadamard",
    }
    config["quantization_config"] = quantization_config
    # Deliberately do NOT set config["quantization"], which would trigger mlx-lm's
    # built-in nn.quantize path (turboquant is not a native mx.quantize mode).
    config.pop("quantization", None)

    print(f"[turboquant] saving to {mlx_path}")
    save(mlx_path, hf_path, model, tokenizer, config)

    if upload_repo is not None:
        from mlx_lm.utils import upload_to_hub

        upload_to_hub(str(mlx_path), upload_repo, hf_path)

    print("[turboquant] done")
    return mlx_path
