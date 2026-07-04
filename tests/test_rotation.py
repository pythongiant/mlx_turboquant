import mlx.core as mx
import numpy as np
import pytest

from mlx_turboquant.rotation import rht, rotate_rows, supported_hadamard_block


# Mix of power-of-two sizes and awkward LLM dims (11008 = down_proj input,
# 14336 = 28*512, 4608, 1536) that are not themselves Hadamard sizes.
DIMS = [64, 128, 256, 4096, 1536, 4608, 11008, 14336]


@pytest.mark.parametrize("n", DIMS)
def test_block_divides_dim(n):
    h = supported_hadamard_block(n)
    assert n % h == 0
    assert h >= 1


@pytest.mark.parametrize("n", DIMS)
def test_rotation_is_orthogonal(n):
    # An orthogonal transform preserves the Euclidean norm of every vector.
    # (RHT is orthogonal, not self-inverse — the invariance we actually use is
    # tested in test_matmul_invariance / preservation of inner products.)
    mx.random.seed(0)
    x = mx.random.normal((3, n))
    y = rht(x, seed=1234)
    nx = mx.sqrt(mx.sum(x * x, axis=-1))
    ny = mx.sqrt(mx.sum(y * y, axis=-1))
    assert mx.allclose(nx, ny, rtol=1e-3, atol=1e-3).item()


@pytest.mark.parametrize("n", DIMS)
def test_rotation_preserves_inner_products(n):
    # The KV-cache path relies on <rht(q), rht(k)> == <q, k>.
    mx.random.seed(0)
    q = mx.random.normal((4, n))
    k = mx.random.normal((4, n))
    ref = mx.sum(q * k, axis=-1)
    got = mx.sum(rht(q, seed=5) * rht(k, seed=5), axis=-1)
    denom = mx.maximum(mx.abs(ref), mx.array(1.0))
    assert (mx.abs(ref - got) / denom).max().item() < 1e-2


@pytest.mark.parametrize("n", DIMS)
def test_matmul_invariance(n):
    # The whole point: rotating BOTH weight rows and the activation leaves the
    # linear output unchanged, so (W R)(Rᵀ x) == W x.
    mx.random.seed(0)
    out_features = 32
    w = mx.random.normal((out_features, n))
    x = mx.random.normal((5, n))
    seed = 99
    w_rot = rotate_rows(w, seed=seed)
    x_rot = rht(x, seed=seed)
    ref = x @ w.T
    got = x_rot @ w_rot.T
    # Relative error (values scale like sqrt(n), so use an rms-relative bound).
    rel = (mx.abs(ref - got).max() / mx.sqrt(mx.mean(ref * ref))).item()
    assert rel < 1e-3, rel


@pytest.mark.parametrize("n", [4096, 11008])
def test_rotation_whitens_outliers(n):
    # A weight row with a big outlier should have its dynamic range (max/rms)
    # substantially reduced after rotation — this is why quantization improves.
    mx.random.seed(0)
    w = mx.random.normal((1, n))
    w[0, 0] = 40.0  # inject an outlier
    before = (mx.abs(w).max() / mx.sqrt(mx.mean(w * w))).item()
    wr = rotate_rows(w, seed=7)
    after = (mx.abs(wr).max() / mx.sqrt(mx.mean(wr * wr))).item()
    assert after < before / 3, (before, after)
