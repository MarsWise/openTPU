"""Raw unsigned input conversion, including padding, NaN-like bits and chunking."""
import dataclasses

import numpy as np
import pytest

from opentpu import Config, isa as I, rtlsim
from opentpu.isasim import Machine, SimError
from opentpu.kernels.u8_input import program
from opentpu.target import arch_hash


def case(count, waitc=False):
    cfg = Config(D=128, MCOLS=8, ACT_BLOCKS=16, TMEM_WORDS=1024, DRAM_BYTES=1 << 20,
                 VU8=True, SEQ_WC=waitc)
    rng = np.random.default_rng(28 + count)
    x = rng.integers(0, 256, size=count, dtype=np.uint8)
    # Test every value plus packed patterns which are NaNs or subnormals as fp32.
    if count >= 256:
        x[:256] = np.arange(256, dtype=np.uint8)
    if count >= 8:
        x[:8] = np.array([1, 0, 0xc0, 0x7f, 1, 0, 0, 0], np.uint8)
    if count == 1024:
        x = np.repeat(np.arange(256, dtype=np.uint8), 4)
    image = np.zeros(cfg.DRAM_BYTES, np.uint8)
    image[:count] = x
    image[0x20000:0x20000 + 4 * count + 8] = 0x99
    return cfg, x, image, program(cfg, 0, 0x20000, count)


@pytest.mark.parametrize('count', [1, 3, 4, 5, 257, 1001, 1024])
def test_u8_matches_exact_float_and_keeps_tail(count):
    cfg, x, image, prog = case(count)
    s = Machine(cfg, [prog], [image]).run().slices[0]
    np.testing.assert_array_equal(s.dram[0x20000:0x20000 + 4 * count].copy().view(np.float32),
                                  x.astype(np.float32))
    np.testing.assert_array_equal(s.dram[0x20000 + 4 * count:0x20000 + 4 * count + 8],
                                  np.full(8, 0x99, np.uint8))


@pytest.mark.parametrize('count,waitc', [(1, False), (257, True), (1001, False), (1024, True)])
def test_u8_rtl_matches_isa_with_axi_stalls(have_verilator, count, waitc):
    cfg, x, image, prog = case(count, waitc)
    ref = Machine(cfg, [prog], [image.copy()]).run().slices[0]
    drams, tmems, _ = rtlsim.run(cfg, [prog], [image.copy()], axi=True, stall=35, seed=count + 3)
    np.testing.assert_array_equal(tmems[0], ref.tmem)
    np.testing.assert_array_equal(drams[0], ref.dram)


def test_u8_capability_and_input_validation():
    from opentpu.host.board import Board, ConfigMismatch, device_config
    from opentpu.host.fake import FakeTransport
    cfg, _, image, prog = case(5)
    old = dataclasses.replace(cfg, VU8=False)
    assert arch_hash(old) != arch_hash(cfg)
    with pytest.raises(SimError, match='Config.VU8'):
        Machine(old, [prog], [image]).run()
    with pytest.raises(ValueError, match='Config.VU8'):
        program(old, 0, 0x20000, 5)
    for args in [(1, 0x20000, 5), (0, 0, 5), (0, 0x20000, 0)]:
        with pytest.raises(ValueError):
            program(cfg, *args)
    code = I.assemble(prog)
    with pytest.raises(ConfigMismatch, match='VU8'):
        Board(FakeTransport(devname=None, regmap=5)).load_program(0x1000, code)
    board = Board(FakeTransport(devname=None, regmap=5, vu8=True))
    assert device_config(board.info()).VU8
    board.load_program(0x1000, code)
