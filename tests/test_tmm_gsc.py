"""Group scales must multiply block sums before accumulation, including packed quarters."""
import dataclasses

import numpy as np
import pytest

from opentpu import isa as I
from opentpu.isasim import Config, Machine, SimError
from opentpu.quant import pack_ternary_mxu


def case(kb=5, m=6, acc=False):
    cfg = Config(S=1, D=128, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=16, TGSC=True)
    rng = np.random.default_rng(1800 + kb + m)
    n, ors, out = 7, 9, 8192
    x = rng.normal(size=(m, kb * cfg.D)).astype(np.float32)
    w = rng.integers(-1, 2, (n, kb * cfg.D), dtype=np.int8)
    packed = pack_ternary_mxu(w, cfg.D)
    # Deliberately unrelated group scales; a per-row scale cannot reproduce this product.
    scales = rng.uniform(.001, .2, (n, kb)).astype(np.float16).astype(np.float32)
    image = np.zeros(cfg.DRAM_BYTES, np.uint8)
    image[:x.nbytes] = x.view(np.uint8).reshape(-1)
    image[0x20000:0x20000 + packed.size] = packed.reshape(-1)
    image[0x40000:0x40000 + scales.nbytes] = scales.view(np.uint8).reshape(-1)
    initial = rng.normal(size=(m, ors)).astype(np.float32)
    image[0x50000:0x50000 + initial.nbytes] = initial.view(np.uint8).reshape(-1)
    prog = [I.ld(0, 0, x.size), I.qact(0, m, 0, kb, kb * cfg.D)]
    if acc:
        prog.append(I.ld(0x50000, out, initial.size))
    prog += [I.tmm(0x20000, 0x40000, out, n, kb, packed.shape[1], ors, m, 0,
                   4 * kb, gsc=True, acc=acc, rmax=m <= cfg.MCOLS), I.halt()]
    return cfg, image, prog, x, w, scales, initial, out, ors


@pytest.mark.parametrize("kb,m,acc", [(1, 1, False), (3, 6, True), (4, 1, True),
                                     (5, 6, False), (7, 1, False)])
def test_gsc_isa_against_float_product(kb, m, acc):
    cfg, image, prog, x, w, s, initial, out, ors = case(kb, m, acc)
    sl = Machine(cfg, [prog], [image]).run().slices[0]
    actual = sl.tmem[out:out + m * ors].view(np.float32).reshape(m, ors)[:, :w.shape[0]]
    expected = sum((x[:, k*128:(k+1)*128] @ w[:, k*128:(k+1)*128].T) * s[:, k]
                   for k in range(kb))
    if acc:
        expected += initial[:, :w.shape[0]]
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < .012
    if m <= cfg.MCOLS:
        np.testing.assert_array_equal(sl.tmem[out + m*ors:out + m*ors + m].view(np.float32),
                                      actual.max(axis=1))


def test_gsc_refuses_old_firmware_and_bad_flags():
    cfg, image, prog, *_ = case()
    with pytest.raises(SimError, match="Config.TGSC"):
        Machine(dataclasses.replace(cfg, TGSC=False), [prog], [image]).run()
    for flags in ({"unit": True}, {"rsc": True}, {"pair": True}, {"quad": True}):
        with pytest.raises(ValueError, match="GSC"):
            I.tmm(0, 0, 0, 2, 4, 128, 2, 1, 0, 16, gsc=True, **flags)
    with pytest.raises(ValueError, match="scale row"):
        I.tmm(0, 0, 0, 2, 4, 128, 2, 1, 0, 4, gsc=True)


@pytest.mark.parametrize("kb,m,acc", [(1, 1, False), (3, 6, True), (4, 1, True),
                                     (5, 6, False), (7, 1, False)])
def test_gsc_rtl(have_verilator, kb, m, acc):
    from opentpu import rtlsim
    cfg, image, prog, *_ = case(kb, m, acc)
    ref = Machine(cfg, [prog], [image.copy()]).run().slices[0]
    drams, tmems, _ = rtlsim.run(cfg, [prog], [image.copy()], axi=True, stall=35, seed=kb+29)
    np.testing.assert_array_equal(tmems[0], ref.tmem)
    np.testing.assert_array_equal(drams[0], ref.dram)


def test_gsc_firmware_capability_is_required_before_program_load():
    from opentpu.host.board import Board, ConfigMismatch, device_config
    from opentpu.host.fake import FakeTransport
    prog = I.assemble([I.tmm(0, 0, 0, 2, 4, 128, 2, 1, 0, 16, gsc=True), I.halt()])
    with pytest.raises(ConfigMismatch, match='TGSC'):
        Board(FakeTransport(regmap=5, devname=None)).load_program(0x1000, prog)
    transport = FakeTransport(regmap=5, devname=None, tgsc=True)
    board = Board(transport)
    assert device_config(board.info()).TGSC
    board.load_program(0x1000, prog)


@pytest.mark.parametrize('waitc', [False, True])
def test_gsc_and_legacy_commands_share_one_completion_stream_rtl(have_verilator, waitc):
    from opentpu import rtlsim
    cfg, image, prog, *_ = case(kb=5, m=1)
    cfg = dataclasses.replace(cfg, SEQ_WC=waitc)
    legacy = I.tmm(0x20000, 0x40000, 10000, 7, 5, 256, 9, 1, 0, 20, rsc=True)
    gsc = prog[-2]
    prog = prog[:-2] + [legacy, legacy, gsc, legacy, gsc, I.halt()]
    ref = Machine(cfg, [prog], [image.copy()]).run().slices[0]
    drams, tmems, _ = rtlsim.run(cfg, [prog], [image.copy()], axi=True, stall=45, seed=63)
    np.testing.assert_array_equal(tmems[0], ref.tmem)
    np.testing.assert_array_equal(drams[0], ref.dram)
