"""Monkeypatch hooks so vanilla ``mlx_lm`` transparently handles TurboQuant dirs.

``register()`` wraps ``mlx_lm.utils.load_model``; for a ``turboquant`` model
directory it delegates to our self-contained loader, and for everything else it
calls the original untouched.  After ``register()`` you can use the ordinary
``mlx_lm.load`` / ``mlx_lm.generate`` / ``mlx_lm.server`` entry points.
"""

from __future__ import annotations

_REGISTERED = False


def register() -> None:
    global _REGISTERED

    # Attention patch must run even if load_model was already wrapped, and it is
    # cheap + idempotent, so do it every call (new model modules may have been
    # imported since the last register()).
    from .attention import patch_attention

    patch_attention()

    if _REGISTERED:
        return

    import mlx_lm.utils as U

    from .loader import _turboquant_load_model, is_turboquant_dir

    _orig_load_model = U.load_model

    def load_model(model_path, lazy=False, strict=True, model_config=None, **kw):
        try:
            config = U.load_config(model_path)
        except Exception:
            config = {}
        if is_turboquant_dir(config):
            return _turboquant_load_model(
                model_path, lazy=lazy, strict=strict, model_config=model_config
            )
        return _orig_load_model(
            model_path, lazy=lazy, strict=strict, model_config=model_config, **kw
        )

    load_model._turboquant_wrapped = True  # idempotency marker
    if not getattr(U.load_model, "_turboquant_wrapped", False):
        U.load_model = load_model

    _REGISTERED = True
