import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_turboquant import qjl
from mlx_turboquant.rotation import rht, supported_hadamard_block


def test_pack_unpack_signs_roundtrip():
    mx.random.seed(0)
    x = mx.random.normal((4, 128))
    packed = qjl.pack_signs(x)
    assert packed.shape == (4, 128 // 32)
    pm1 = qjl.unpack_signs_pm1(packed, 128)
    assert mx.array_equal(pm1, mx.where(x >= 0, 1.0, -1.0)).item()


@pytest.mark.parametrize("d", [64, 128])
def test_qjl_estimator_is_unbiased(d):
    # Over many random (query, residual) pairs, the mean QJL estimate of <q, r>
    # should match the true value (unbiasedness), with the constant sqrt(pi/2d).
    mx.random.seed(0)
    N = 4000
    q = mx.random.normal((N, d))
    r = mx.random.normal((N, d))
    true = mx.sum(q * r, axis=-1)

    # Sketch the residual; project the query through the same JL map.
    rnorm, packed = qjl.sketch_residual(r, seed=qjl.QJL_SEED)
    qp = qjl.project_query(q, seed=qjl.QJL_SEED)
    pm1 = qjl.unpack_signs_pm1(packed, d)
    est = qjl.qjl_constant(d) * rnorm[..., 0] * mx.sum(qp * pm1, axis=-1)

    true_np = np.array(true)
    est_np = np.array(est)
    # Unbiased: mean of (est - true) ~ 0 relative to signal scale.
    bias = (est_np - true_np).mean()
    signal = np.sqrt((true_np**2).mean())
    assert abs(bias) < 0.15 * signal, (bias, signal)
    # And positively correlated with the truth (it is a real estimator).
    corr = np.corrcoef(est_np, true_np)[0, 1]
    assert corr > 0.3, corr


@pytest.mark.parametrize("d", [64, 128])
def test_qjl_reduces_inner_product_error_vs_mse_only(d):
    # End-to-end claim: <Rq, k̂> + QJL(<Rq, r>) is a better estimate of <q, k>
    # than <Rq, k̂> alone (the MSE-only, biased estimate), at low bits.
    from mlx_turboquant import codebook as cb

    mx.random.seed(0)
    bits, gs, N = 2, 64, 2000
    block = supported_hadamard_block(d)
    q = mx.random.normal((N, d))
    k = mx.random.normal((N, d))
    true = mx.sum(q * k, axis=-1)  # rotation preserves <q,k>

    Rq = rht(q, 0xC0FFEE, block)
    Rk = rht(k, 0xC0FFEE, block)
    p = mx.quantize(Rk, group_size=gs, bits=bits)
    khat = mx.dequantize(*p, group_size=gs, bits=bits)
    r = Rk - khat

    mse_only = mx.sum(Rq * khat, axis=-1)

    rnorm, packed = qjl.sketch_residual(r)
    qp = qjl.project_query(Rq)
    corr = qjl.qjl_constant(d) * rnorm[..., 0] * mx.sum(qp * qjl.unpack_signs_pm1(packed, d), axis=-1)
    with_qjl = mse_only + corr

    def rel(est):
        return (mx.sqrt(mx.mean((true - est) ** 2)) / mx.sqrt(mx.mean(true * true))).item()

    assert rel(with_qjl) < rel(mse_only), (rel(with_qjl), rel(mse_only))


@pytest.mark.parametrize("bits", [2, 3])
def test_qjl_removes_inner_product_bias_for_correlated_pairs(bits):
    # The paper's actual motivation: MSE quantizers are *biased* for inner-product
    # estimation, and the bias is worst for high-similarity (query, key) pairs —
    # exactly the ones attention softmax weights most. With correlated q, k the
    # MSE-only estimate systematically under-counts; QJL removes that bias.
    from mlx_turboquant import codebook as cb  # noqa: F401

    mx.random.seed(0)
    d, gs, N, rho = 128, 64, 8000, 0.9
    block = supported_hadamard_block(d)
    k = mx.random.normal((N, d))
    q = rho * k + math.sqrt(1 - rho**2) * mx.random.normal((N, d))  # correlated
    true = mx.sum(q * k, axis=-1)

    Rq, Rk = rht(q, 0xC0FFEE, block), rht(k, 0xC0FFEE, block)
    p = mx.quantize(Rk, group_size=gs, bits=bits)
    khat = mx.dequantize(*p, group_size=gs, bits=bits)
    r = Rk - khat

    mse_only = mx.sum(Rq * khat, axis=-1)
    rn, pk = qjl.sketch_residual(r)
    corr = qjl.qjl_constant(d) * rn[..., 0] * mx.sum(
        qjl.project_query(Rq) * qjl.unpack_signs_pm1(pk, d), axis=-1
    )
    with_qjl = mse_only + corr

    scale = mx.sqrt(mx.mean(true * true))
    bias_mse = (mx.mean(mse_only - true) / scale).item()
    bias_qjl = (mx.mean(with_qjl - true) / scale).item()
    assert bias_mse < -0.01, bias_mse            # MSE-only is meaningfully biased
    assert abs(bias_qjl) < abs(bias_mse) * 0.5   # QJL at least halves the bias
