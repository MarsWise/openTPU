import numpy as np
import pytest

from opentpu import Config
from opentpu.isasim import Machine
from opentpu.kernels.image_input import patchify_programs


@pytest.mark.parametrize('channels,patch,height,width', [(3, 2, 4, 6), (1, 3, 6, 9),
                                                        (4, 1, 3, 4), (3, 14, 14, 28)])
def test_patch_rows_match_independent_reference(channels, patch, height, width):
    cfg = Config(D=128, MCOLS=8, ACT_BLOCKS=16, TMEM_WORDS=1024,
                 IMEM_WORDS=512, DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(channels + patch)
    x = rng.standard_normal((height, width, channels)).astype(np.float32)
    img = np.zeros(cfg.DRAM_BYTES, np.uint8)
    img[:x.nbytes] = x.reshape(-1).view(np.uint8)
    dst = 0x40000
    img[dst:dst + x.nbytes + 16] = 0x99
    programs = patchify_programs(cfg, 0, dst, height, width, patch, channels)
    for code in programs:
        assert len(code) <= cfg.IMEM_WORDS // 8
        img = Machine(cfg, [code], [img]).run().slices[0].dram
    actual = img[dst:dst + x.nbytes].copy().view(np.float32)
    expected = x.reshape(height // patch, patch, width // patch, patch, channels)
    expected = expected.transpose(0, 2, 4, 1, 3).reshape(-1)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(img[dst + x.nbytes:dst + x.nbytes + 16],
                                  np.full(16, 0x99, np.uint8))


def test_patch_validation():
    cfg = Config(TMEM_WORDS=1024, IMEM_WORDS=512, DRAM_BYTES=1 << 20)
    with pytest.raises(ValueError):
        patchify_programs(cfg, 0, 0x40000, 5, 6, 2)
    with pytest.raises(ValueError):
        patchify_programs(cfg, 0, 0, 4, 4, 2)
    with pytest.raises(ValueError):
        patchify_programs(cfg, 0, 0x40000, 6, 6, 2, order='minicpm')


def test_minicpm_window_order():
    cfg = Config(D=128, MCOLS=8, ACT_BLOCKS=16, TMEM_WORDS=1024,
                 IMEM_WORDS=512, DRAM_BYTES=1 << 20)
    x = np.arange(8 * 16 * 3, dtype=np.float32).reshape(8, 16, 3)
    image = np.zeros(cfg.DRAM_BYTES, np.uint8)
    image[:x.nbytes] = x.reshape(-1).view(np.uint8)
    programs = patchify_programs(cfg, 0, 0x40000, 8, 16, 2, 3, order='minicpm')
    for code in programs:
        image = Machine(cfg, [code], [image]).run().slices[0].dram
    actual = image[0x40000:0x40000 + x.nbytes].copy().view(np.float32).reshape(-1, 12)
    raster = x.reshape(4, 2, 8, 2, 3).transpose(0, 2, 4, 1, 3).reshape(32, 12)
    order = np.arange(32).reshape(1, 2, 2, 2, 2, 2).transpose(0, 3, 1, 4, 2, 5).reshape(-1)
    np.testing.assert_array_equal(actual, raster[order])


def test_padded_patch_rows_preserve_padding():
    cfg = Config(D=128, MCOLS=8, ACT_BLOCKS=16, TMEM_WORDS=1024,
                 IMEM_WORDS=512, DRAM_BYTES=1 << 20)
    x = np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3)
    image = np.zeros(cfg.DRAM_BYTES, np.uint8)
    image[:x.nbytes] = x.reshape(-1).view(np.uint8)
    dst, stride = 0x40000, 16
    image[dst:dst + 4 * 4 * stride] = 0x99
    for code in patchify_programs(cfg, 0, dst, 4, 4, 2, 3, dst_stride=stride):
        image = Machine(cfg, [code], [image]).run().slices[0].dram
    actual = image[dst:dst + 4 * 4 * stride].reshape(4, stride * 4)
    expected = x.reshape(2, 2, 2, 2, 3).transpose(0, 2, 4, 1, 3).reshape(4, 12)
    np.testing.assert_array_equal(actual[:, :48].copy().view(np.float32), expected)
    np.testing.assert_array_equal(actual[:, 48:], np.full((4, 16), 0x99, np.uint8))


def test_u8_normalization_then_patchify_matches_minicpm_layout():
    from opentpu.kernels.u8_input import program_normalize
    cfg = Config(D=128, MCOLS=8, ACT_BLOCKS=16, TMEM_WORDS=1024,
                 IMEM_WORDS=512, DRAM_BYTES=1 << 20, VU8=True, SEQ_WC=True)
    rng = np.random.default_rng(17)
    raw = rng.integers(0, 256, (14, 28, 3), dtype=np.uint8)
    img = np.zeros(cfg.DRAM_BYTES, np.uint8)
    img[:raw.size] = raw.reshape(-1)
    normalize = program_normalize(cfg, 0, 0x20000, 14 * 28,
                                  (1 / 127.5,) * 3, (-1.0,) * 3)
    programs = [normalize] + patchify_programs(cfg, 0x20000, 0x40000, 14, 28, 14, 3)
    for code in programs:
        img = Machine(cfg, [code], [img]).run().slices[0].dram
    actual = img[0x40000:0x40000 + raw.size * 4].copy().view(np.float32)
    # MiniCPM-V's existing processor: per-patch (channel, row, column).
    reference = (raw.astype(np.float32) - np.float32(127.5)) / np.float32(127.5)
    reference = reference.reshape(1, 14, 2, 14, 3).transpose(0, 2, 4, 1, 3).reshape(-1)
    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-7)
