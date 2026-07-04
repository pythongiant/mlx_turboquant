"""mlx-turboquant — TurboQuant rotation quantization for MLX-LM.

Public API (lazy so that ``import mlx_turboquant`` stays light and does not pull
in ``mlx_lm`` until you actually convert / load a model)::

    import mlx_turboquant as tq
    tq.convert("mlx-community/Llama-3.2-3B-Instruct", "./tq", bits=4)   # quantize
    model, tokenizer = tq.load("./tq")                                  # run
    tq.register()   # make plain mlx_lm.load / mlx_lm.generate understand tq dirs

See https://arxiv.org/pdf/2504.19874 for the algorithm.
"""

from __future__ import annotations

__version__ = "0.0.1"

# Lightweight, dependency-free (mlx-only) building blocks are safe to import now.
from .rotation import rht, rotate_rows, supported_hadamard_block  # noqa: F401
from .codebook import lloyd_max_levels  # noqa: F401

__all__ = [
    "rht",
    "rotate_rows",
    "supported_hadamard_block",
    "lloyd_max_levels",
    "convert",
    "load",
    "make_prompt_cache",
    "register",
]


def __getattr__(name):
    # Defer mlx_lm-dependent entry points until first use.
    if name in ("convert",):
        from .convert import convert

        return convert
    if name in ("load",):
        from .loader import load

        return load
    if name == "make_prompt_cache":
        from .kv_cache import make_prompt_cache

        return make_prompt_cache
    if name == "register":
        from .patch import register

        return register
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
