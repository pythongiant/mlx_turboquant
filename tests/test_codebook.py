import mlx.core as mx
import numpy as np
import pytest

from mlx_turboquant import codebook as cb


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_levels_count_and_symmetry(bits):
    levels = np.array(cb.lloyd_max_levels(bits))
    assert len(levels) == (1 << bits)
    assert np.all(np.diff(levels) > 0)  # sorted, distinct
    # Gaussian is symmetric → levels should be (approximately) antisymmetric.
    assert np.allclose(levels, -levels[::-1], atol=1e-6)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_lloyd_max_beats_uniform_on_gaussian(bits):
    # The defining property: Lloyd–Max MSE < a matched-range uniform quantizer's
    # MSE on the Gaussian source. This is the whole reason we ship a codebook
    # rather than reuse affine (uniform) quantization.
    xs, w = cb._gaussian_grid()
    levels = np.array(cb.lloyd_max_levels(bits))

    # Lloyd–Max distortion.
    idx = np.abs(xs[:, None] - levels[None, :]).argmin(axis=1)
    mse_lm = (w * (xs - levels[idx]) ** 2).sum()

    # Uniform quantizer spanning the same [min, max] with 2**bits levels.
    lo, hi = levels[0], levels[-1]
    ulevels = np.linspace(lo, hi, 1 << bits)
    uidx = np.abs(xs[:, None] - ulevels[None, :]).argmin(axis=1)
    mse_uniform = (w * (xs - ulevels[uidx]) ** 2).sum()

    assert mse_lm < mse_uniform, (bits, mse_lm, mse_uniform)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_quantize_dequantize_roundtrip(bits):
    mx.random.seed(0)
    group_size = 64
    x = mx.random.normal((8, group_size)) * 2.5  # arbitrary per-row scale
    scale = cb.group_scale(x, group_size)
    idx = cb.quantize_group(x.reshape(8, 1, group_size), bits, scale)
    xr = cb.dequantize_indices(idx, bits, scale).reshape(8, group_size)
    # Reconstruction should be much closer than the signal magnitude.
    rel = (mx.sqrt(mx.mean((x - xr) ** 2)) / mx.sqrt(mx.mean(x * x))).item()
    assert rel < 0.35 / (bits - 1 + 1e-9) + 0.05, rel
    assert int(idx.max()) < (1 << bits)
    assert int(idx.min()) >= 0
