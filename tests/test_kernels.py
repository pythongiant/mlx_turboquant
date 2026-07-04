import mlx.core as mx
import pytest

from mlx_turboquant import codebook as cb
from mlx_turboquant.kernels.qmm import supports_lut_bits, turbo_qmm


def _pack_and_ref(w, bits, group_size):
    packed, scales = cb.quantize_weight_lut(w, bits, group_size)
    w_hat = cb.dequantize_weight_lut(packed, scales, bits, group_size, w.shape[1])
    return packed, scales, w_hat


@pytest.mark.parametrize("bits", [4, 2, 8])
@pytest.mark.parametrize("shape", [(256, 512), (128, 1024), (64, 128)])
def test_lut_kernel_matches_reference(bits, shape):
    assert supports_lut_bits(bits)
    mx.random.seed(0)
    out_f, in_f = shape
    group_size = 64
    w = mx.random.normal(shape)
    x = mx.random.normal((5, in_f)).astype(mx.float16)

    packed, scales, w_hat = _pack_and_ref(w, bits, group_size)
    ref = x @ w_hat.T  # pure-mx reference using the dequantized weights

    levels = cb.levels_mx(bits)
    got = turbo_qmm(x, packed, scales, levels, bits, group_size, out_f, in_f)

    mx.eval(ref, got)
    rel = (mx.sqrt(mx.mean((ref - got) ** 2)) / mx.sqrt(mx.mean(ref * ref))).item()
    assert rel < 2e-3, rel  # only fp16 accumulation noise should differ


def test_lut_kernel_3d_input():
    mx.random.seed(0)
    out_f, in_f, group_size, bits = 128, 256, 64, 4
    w = mx.random.normal((out_f, in_f))
    x = mx.random.normal((2, 3, in_f)).astype(mx.float16)
    packed, scales, w_hat = _pack_and_ref(w, bits, group_size)
    ref = x @ w_hat.T
    got = turbo_qmm(x, packed, scales, cb.levels_mx(bits), bits, group_size, out_f, in_f)
    assert got.shape == (2, 3, out_f)
    mx.eval(ref, got)
    rel = (mx.sqrt(mx.mean((ref - got) ** 2)) / mx.sqrt(mx.mean(ref * ref))).item()
    assert rel < 2e-3, rel


@pytest.mark.parametrize("bits", [4, 2])
def test_lut_beats_affine_at_low_bits(bits):
    # TurboQuant's core claim, at the linear-layer level: for a (post-rotation)
    # near-Gaussian weight distribution, the optimally-scaled Lloyd–Max LUT
    # reconstructs weights with lower MSE than affine at equal bits/group. The
    # win is concentrated at low bits (levels are scarce, so non-uniform
    # placement matters); at >=8 bits uniform affine is already near-optimal.
    mx.random.seed(0)
    out_f, in_f, group_size = 256, 1024, 64
    w = mx.random.normal((out_f, in_f))

    p, s = cb.quantize_weight_lut(w, bits, group_size)
    w_hat_lut = cb.dequantize_weight_lut(p, s, bits, group_size, in_f)

    aff = mx.quantize(w, group_size=group_size, bits=bits, mode="affine")
    w_hat_aff = mx.dequantize(
        aff[0], aff[1], aff[2], group_size=group_size, bits=bits, mode="affine"
    )

    def rel(a):
        return (mx.sqrt(mx.mean((w - a) ** 2)) / mx.sqrt(mx.mean(w * w))).item()

    assert rel(w_hat_lut) < rel(w_hat_aff), (bits, rel(w_hat_lut), rel(w_hat_aff))
