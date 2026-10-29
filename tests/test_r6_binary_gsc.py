"""BIN group scales: compare identical signs with ternary, including chunk boundaries."""
import dataclasses

import numpy as np
import pytest

from opentpu import Config, isa as I, rtlsim
from opentpu.compiler import Builder, CompileError, QTensor
from opentpu.isasim import Machine, SimError
from opentpu.quant import pack_binary_mxu, pack_ternary_mxu
from opentpu.target import arch_hash


def case(kb, rows, acc=False, waitc=False):
    cfg = Config(D=128, MCOLS=8, ACT_ROWS=8, ACT_BLOCKS=16, DRAM_BYTES=4 << 20,
                 TBIN=True, TGSC=True, TBGSC=True, SEQ_WC=waitc)
    rng = np.random.default_rng(429 + kb + rows)
    n, ors = 5, 7
    x = rng.normal(size=(rows, kb * cfg.D)).astype(np.float32)
    w = rng.choice(np.array([-1, 1], np.int8), (n, kb * cfg.D))
    wb, wt = pack_binary_mxu(w, cfg.D), pack_ternary_mxu(w, cfg.D)
    scales = rng.uniform(.01, .5, (n, kb)).astype(np.float32)
    initial = rng.normal(size=(rows, ors)).astype(np.float32)
    image = np.zeros(cfg.DRAM_BYTES, np.uint8)
    for addr, array in ((0, x), (0x40000, wb), (0x50000, wt),
                        (0x60000, scales), (0x70000, initial)):
        image[addr:addr + array.nbytes] = array.view(np.uint8).reshape(-1)
    prog = [I.ld(0, 0, x.size), I.qact(0, rows, 0, kb, kb * cfg.D)]
    if acc:
        prog += [I.ld(0x70000, 4096, initial.size), I.ld(0x70000, 8192, initial.size)]
    for addr, packed, out, binary in ((0x40000, wb, 4096, True),
                                      (0x50000, wt, 8192, False)):
        prog.append(I.tmm(addr, 0x60000, out, n, kb, packed.shape[1], ors, rows, 0,
                          4 * kb, binary=binary, gsc=True, acc=acc, rmax=True))
    prog.append(I.halt())
    return cfg, image, prog, x, w, scales, initial


@pytest.mark.parametrize('kb,rows,acc', [(1, 1, False), (4, 2, True), (7, 1, False),
                                         (8, 2, True), (9, 1, False), (16, 2, True)])
def test_binary_gsc_isa_matches_ternary_and_float(kb, rows, acc):
    cfg, image, prog, x, w, scales, initial = case(kb, rows, acc)
    sim = Machine(cfg, [prog], [image]).run().slices[0]
    count = rows * 7 + rows
    np.testing.assert_array_equal(sim.tmem[4096:4096 + count], sim.tmem[8192:8192 + count])
    y = sim.tmem[4096:4096 + rows * 7].view(np.float32).reshape(rows, 7)[:, :5]
    expected = sum((x[:, k*128:(k+1)*128] @ w[:, k*128:(k+1)*128].T) * scales[:, k]
                   for k in range(kb))
    if acc:
        expected += initial[:, :5]
    assert np.linalg.norm(y - expected) / np.linalg.norm(expected) < .02


@pytest.mark.parametrize('kb,acc,waitc', [(1, False, False), (7, True, False),
                                           (8, False, True), (9, True, True), (16, False, False)])
def test_binary_gsc_rtl_matches_isa_with_axi_stalls(have_verilator, kb, acc, waitc):
    cfg, image, prog, *_ = case(kb, 2, acc, waitc)
    ref = Machine(cfg, [prog], [image.copy()]).run().slices[0]
    drams, tmems, _ = rtlsim.run(cfg, [prog], [image.copy()], axi=True, stall=35, seed=kb + 9)
    np.testing.assert_array_equal(tmems[0], ref.tmem)
    np.testing.assert_array_equal(drams[0], ref.dram)


def test_binary_gsc_capability_and_compiler():
    from opentpu.host.board import Board, ConfigMismatch, device_config
    from opentpu.host.fake import FakeTransport
    cfg, image, prog, *_ = case(9, 1)
    old = dataclasses.replace(cfg, TBGSC=False)
    with pytest.raises(SimError, match='TBGSC'):
        Machine(old, [prog], [image]).run()
    assert arch_hash(cfg) != arch_hash(old)
    weight = QTensor.binary_weight(0x40000, 0x60000, 5, 9 * 128, 128, srs=36, gsc=True)
    b = Builder(cfg, 0)
    b.dot(b.alloc((1, 9 * 128)), weight)
    instr = next(i for i in b.root if i.op == I.TMM)
    assert instr.w[3] & I.TMM_BIN and instr.flags & I.F_GSC
    with pytest.raises(CompileError, match='TBGSC'):
        b = Builder(old, 0)
        b.dot(b.alloc((1, 9 * 128)), weight)
    code = I.assemble([prog[-3], I.halt()])
    with pytest.raises(ConfigMismatch, match='TBGSC'):
        Board(FakeTransport(devname=None, regmap=5, tbin=True, tgsc=True)).load_program(0x1000, code)
    board = Board(FakeTransport(devname=None, regmap=5, tbin=True, tgsc=True, tbgsc=True))
    assert device_config(board.info()).TBGSC
    board.load_program(0x1000, code)


def test_rtl_still_rejects_reserved_ternary_code_after_binary(have_verilator):
    cfg, image, prog, *_ = case(9, 1)
    image[0x50000] |= np.uint8(3)
    with pytest.raises(SimError, match='reserved ternary code'):
        Machine(cfg, [prog], [image.copy()]).run()
    with pytest.raises(RuntimeError, match='reserved TMM symbol'):
        rtlsim.run(cfg, [prog], [image.copy()], axi=True, stall=35, seed=18)
