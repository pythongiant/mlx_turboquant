import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_turboquant.quant_linear import TurboQuantLinear, seed_for


def _rel_err(a, b):
    return (mx.sqrt(mx.mean((a - b) ** 2)) / mx.sqrt(mx.mean(a * a))).item()


def _weight_mse(W, bits, group_size=64):
    p = mx.quantize(W, group_size=group_size, bits=bits, mode="affine")
    Wh = mx.dequantize(p[0], p[1], p[2], group_size=group_size, bits=bits, mode="affine")
    return _rel_err(W, Wh)


@pytest.mark.parametrize("bits", [4, 3, 2])
def test_from_linear_approximates_fp(bits):
    # Sanity: the quantized linear tracks the fp linear (output rel-err is on the
    # order of the per-weight affine quant error for Gaussian weights).
    mx.random.seed(0)
    lin = nn.Linear(512, 256, bias=True)
    x = mx.random.normal((4, 512))
    ref = lin(x)
    tq = TurboQuantLinear.from_linear(lin, bits=bits, group_size=64, seed=123)
    err = _rel_err(ref, tq(x))
    tol = {4: 0.13, 3: 0.24, 2: 0.55}[bits]
    assert err < tol, (bits, err)


@pytest.mark.parametrize("bits", [4, 3])
def test_rotation_reduces_weight_mse_on_heavy_tailed_weights(bits):
    # The real Phase-1 win: rotation Gaussianizes each group so a single spike no
    # longer inflates the group's uniform step -> lower weight reconstruction MSE.
    # (Measured as weight MSE, the objective TurboQuant's MSE regime optimizes.)
    mx.random.seed(0)
    outf, inf = 256, 1024
    base = mx.random.normal((outf, inf))
    mask = (mx.random.uniform(shape=(outf, inf)) < 0.03).astype(mx.float32)
    W = base + mask * mx.random.normal((outf, inf)) * 10.0  # sparse large spikes

    from mlx_turboquant.rotation import rotate_rows, supported_hadamard_block

    Wr = rotate_rows(W, seed=7, block=supported_hadamard_block(inf))
    mse_affine = _weight_mse(W, bits)
    mse_rot = _weight_mse(Wr, bits)
    assert mse_rot < mse_affine, (bits, mse_rot, mse_affine)


def test_save_load_roundtrip_shapes():
    # The load-time placeholder shells must have identical param shapes/names to
    # the convert-time module so model.load_weights fills them exactly.
    mx.random.seed(0)
    lin = nn.Linear(512, 256, bias=True)
    seed = seed_for("blk.0.q_proj", 0)
    src = TurboQuantLinear.from_linear(lin, bits=4, group_size=64, seed=seed)

    dst = TurboQuantLinear(
        512, 256, bits=4, group_size=64, seed=seed,
        block=src.block, bias=True, mode="affine",
    )
    # Same keys, same shapes.
    for k in ("weight", "scales", "biases", "bias"):
        assert k in src and k in dst
        assert src[k].shape == dst[k].shape

    # Copy params over (simulating load_weights) and check identical output.
    dst.update({k: src[k] for k in ("weight", "scales", "biases", "bias")})
    x = mx.random.normal((3, 512))
    assert mx.allclose(src(x), dst(x)).item()


def test_seed_is_stable_across_calls():
    assert seed_for("model.layers.3.mlp.down_proj", 42) == seed_for(
        "model.layers.3.mlp.down_proj", 42
    )
    assert seed_for("a", 0) != seed_for("b", 0)
