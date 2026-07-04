"""Loading TurboQuant model directories.

``_turboquant_load_model`` mirrors ``mlx_lm.utils.load_model`` but installs
:class:`TurboQuantLinear` shells (via ``quantize_model_from_config``) instead of
the built-in ``nn.quantize`` path, because TurboQuant is not one of MLX's
built-in ``mx.quantize`` modes.  ``patch.register()`` routes only ``turboquant``
model dirs here; everything else uses the stock loader.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .quant_linear import turboquant_quantize

__all__ = ["load", "quantize_model_from_config", "is_turboquant_dir"]


def quantize_model_from_config(model: nn.Module, config: dict) -> nn.Module:
    """Install TurboQuantLinear placeholder shells according to config."""
    qc = config.get("quantization_config") or config.get("quantization") or {}
    turboquant_quantize(
        model,
        bits=int(qc.get("bits", 4)),
        group_size=int(qc.get("group_size", 64)),
        base_seed=int(qc.get("rot_base_seed", 0)),
        mode=qc.get("mode", "affine"),
        skip=tuple(qc.get("skip", ("lm_head",))),
        from_placeholders=True,
    )
    return model


def is_turboquant_dir(config: dict) -> bool:
    qc = config.get("quantization_config") or {}
    return qc.get("quant_method") == "turboquant"


def _turboquant_load_model(
    model_path: Path,
    lazy: bool = False,
    strict: bool = True,
    model_config: Optional[Dict[str, Any]] = None,
    **_,
) -> Tuple[nn.Module, dict]:
    # Reuse mlx-lm's public building blocks for everything except the swap.
    from mlx_lm.utils import _get_classes, load_config

    config = load_config(model_path)
    if model_config:
        config.update(model_config)

    weight_files = glob.glob(str(model_path / "model*.safetensors"))
    if not weight_files and strict:
        raise FileNotFoundError(f"No safetensors found in {model_path}")
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    model_class, model_args_class = _get_classes(config=config)
    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    quantize_model_from_config(model, config)

    model.eval()
    model.load_weights(list(weights.items()), strict=strict)
    if not lazy:
        mx.eval(model.parameters())
    return model, config


def load(path_or_hf_repo: str, tokenizer_config: Optional[dict] = None, lazy: bool = False):
    """Load a TurboQuant model + tokenizer (works for stock models too)."""
    from mlx_lm.utils import _download, load_config, load_tokenizer

    from .patch import register

    register()  # ensure mlx_lm.load_model routes turboquant dirs here
    model_path = _download(path_or_hf_repo)
    config = load_config(model_path)
    if is_turboquant_dir(config):
        model, config = _turboquant_load_model(model_path, lazy=lazy)
        tokenizer = load_tokenizer(
            model_path, tokenizer_config or {}, eos_token_ids=config.get("eos_token_id")
        )
        return model, tokenizer
    # Fall back to the stock loader for non-turboquant directories.
    from mlx_lm import load as mlx_load

    return mlx_load(path_or_hf_repo, tokenizer_config=tokenizer_config or {}, lazy=lazy)
