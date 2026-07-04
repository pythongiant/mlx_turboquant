import mlx.core as mx
import pytest

from mlx_turboquant.kv_cache import TurboQuantKVCache


def _scores(q, k):
    return q @ k.swapaxes(-1, -2)


def test_rotation_preserves_attention_scores_exactly():
    # <rotate_q(q), rotate_k(k)> == <q, k> (orthogonal rotation), no quant.
    mx.random.seed(0)
    c = TurboQuantKVCache(group_size=64, bits=4)
    c._block(128)  # resolve head_dim block
    q = mx.random.normal((1, 2, 5, 128))
    k = mx.random.normal((1, 2, 7, 128))
    ref = _scores(q, k)
    got = _scores(c.rotate_query(q), c.rotate_key(k))
    rel = (mx.abs(ref - got).max() / mx.sqrt(mx.mean(ref * ref))).item()
    assert rel < 1e-2, rel


@pytest.mark.parametrize("bits", [8, 4])
def test_stored_keys_are_rotated_and_recover_scores(bits):
    # Feed keys/values through the cache; the stored keys are rotated+quantized.
    # Scores from (rotated query) x (dequantized stored keys) must approximate
    # the true <q, k>, AND rotating the query must be *necessary* (using the
    # un-rotated query gives a much worse score) — proving rotation is applied.
    mx.random.seed(0)
    head_dim, gs = 128, 64
    c = TurboQuantKVCache(group_size=gs, bits=bits)
    q = mx.random.normal((1, 1, 4, head_dim))
    k = mx.random.normal((1, 1, 16, head_dim))
    v = mx.random.normal((1, 1, 16, head_dim))

    q_keys, _ = c.update_and_fetch(k, v)
    k_deq = mx.dequantize(*q_keys, group_size=gs, bits=bits)

    ref = _scores(q, k)
    good = _scores(c.rotate_query(q), k_deq)      # correct: query rotated
    bad = _scores(q, k_deq)                        # wrong: query not rotated

    def rel(s):
        return (mx.sqrt(mx.mean((ref - s) ** 2)) / mx.sqrt(mx.mean(ref * ref))).item()

    assert rel(good) < 0.25, rel(good)
    assert rel(good) < rel(bad) * 0.5, (rel(good), rel(bad))


def test_meta_state_roundtrip():
    c = TurboQuantKVCache(group_size=32, bits=4)
    mx.random.seed(0)
    k = mx.random.normal((1, 1, 8, 128))
    c.update_and_fetch(k, k)
    ms = c.meta_state
    c2 = TurboQuantKVCache()
    c2.state = c.state
    c2.meta_state = ms
    assert c2.offset == c.offset
    assert c2.bits == c.bits
    assert c2.group_size == c.group_size
