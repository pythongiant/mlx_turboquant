"""Tests for threaded-server compatibility and the `turboquant serve` KV patch."""

import threading

import mlx.core as mx


def test_rotation_helpers_are_thread_safe():
    # mlx-lm's HTTP server runs each request in a worker thread with no default
    # GPU stream. signs_for (numpy) and supported_hadamard_block (CPU-stream
    # probe) must work there — regression for "no Stream(gpu, 0)" in threads.
    from mlx_turboquant.rotation import signs_for, supported_hadamard_block

    out = {}

    def worker():
        try:
            d = signs_for(4242, 128)          # numpy: no stream needed
            blk = supported_hadamard_block(128)  # CPU-stream probe
            mx.eval(d)                          # force eval inside the thread
            out["result"] = (int(d.shape[0]), blk)
        except Exception as e:  # pragma: no cover
            out["result"] = ("ERROR", str(e))

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert out["result"] == (128, 128), out["result"]


def test_signs_are_deterministic_and_pm1():
    from mlx_turboquant.rotation import signs_for

    a = signs_for(7, 256)
    b = signs_for(7, 256)
    assert mx.array_equal(a, b).item()                    # stable across calls
    assert mx.all((a == 1.0) | (a == -1.0)).item()        # exactly ±1


def test_server_kv_patch_installs_turboquant_cache():
    from mlx_turboquant.cli import _patch_server_kv
    from mlx_turboquant.kv_cache import TurboQuantKVCache

    import mlx_lm.server as server

    orig = server.make_prompt_cache
    try:
        _patch_server_kv(bits=4, group_size=64, qjl=True)

        class _Model:
            layers = [object(), object(), object()]

        cache = server.make_prompt_cache(_Model())
        assert len(cache) == 3
        assert all(isinstance(c, TurboQuantKVCache) for c in cache)
        assert cache[0].qjl and cache[0].bits == 4 and cache[0].group_size == 64
    finally:
        server.make_prompt_cache = orig
