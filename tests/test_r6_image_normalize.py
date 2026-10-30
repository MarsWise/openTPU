"""FPGA uint8 input and per-channel affine normalization."""
import numpy as np
import pytest

from opentpu import Config
from opentpu.isasim import Machine
from opentpu.kernels.u8_input import program_normalize


@pytest.mark.parametrize('pixels,channels', [(1, 1), (5, 3), (87, 3), (67, 4)])
@pytest.mark.parametrize('waitc', [False, True])
def test_normalize_matches_fp32_and_keeps_surroundings(pixels, channels, waitc):
    cfg = Config(D=128, MCOLS=8, ACT_BLOCKS=16, TMEM_WORDS=128,
                 DRAM_BYTES=1 << 20, VU8=True, SEQ_WC=waitc)
    rng = np.random.default_rng(pixels * 31 + channels)
    x = rng.integers(0, 256, size=(pixels, channels), dtype=np.uint8)
    scale = np.asarray([0.125, -0.25, 1.0 / 255, 2.0][:channels], np.float32)
    bias = np.asarray([-1.0, 2.0, -0.5, 3.0][:channels], np.float32)
    img = np.zeros(cfg.DRAM_BYTES, np.uint8)
    img[:x.size] = x.reshape(-1)
    dst = 0x20000
    img[dst:dst + x.size * 4 + 8] = 0x99
    code = program_normalize(cfg, 0, dst, pixels, scale, bias)
    output = Machine(cfg, [code], [img]).run().slices[0].dram
    actual = output[dst:dst + x.size * 4].copy().view(np.float32).reshape(x.shape)
    expected = x.astype(np.float32) * scale + bias
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(output[dst + x.size * 4:dst + x.size * 4 + 8],
                                  np.full(8, 0x99, np.uint8))


def test_normalize_validates_layout_and_constants():
    cfg = Config(TMEM_WORDS=128, DRAM_BYTES=1 << 20, VU8=True)
    with pytest.raises(ValueError):
        program_normalize(cfg, 0, 0x20000, 1, (1.0, 2.0), (0.0, 0.0))
    with pytest.raises(ValueError):
        program_normalize(cfg, 0, 0x20000, 1, (float('nan'),), (0.0,))
    with pytest.raises(ValueError):
        program_normalize(cfg, 0, 0, 1, (1.0,), (0.0,))
