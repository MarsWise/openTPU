"""Binary weights are independent from ternary/GSC and agree with ternary +/-1."""
import numpy as np
import pytest

from opentpu import Config, isa as I, rtlsim
from opentpu.isasim import Machine, SimError
from opentpu.quant import pack_binary_mxu, pack_ternary_mxu
from opentpu.target import arch_hash
from opentpu.compiler import Builder, CompileError, QTensor


def case(kb, unit, cfg=None):
    cfg = cfg or Config(D=32, MCOLS=4, ACT_BLOCKS=16, TBIN=True)
    D, M, N = cfg.D, 2, 5
    rng = np.random.default_rng(2026 + kb + int(unit))
    x = rng.normal(size=(M, kb * D)).astype(np.float32)
    w = rng.choice(np.array([-1, 1], dtype=np.int8), size=(N, kb * D))
    wb, wt = pack_binary_mxu(w, D), pack_ternary_mxu(w, D)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    dram[:x.nbytes] = x.view(np.uint8).reshape(-1)
    dram[0x40000:0x40000 + wb.size] = wb.reshape(-1)
    dram[0x50000:0x50000 + wt.size] = wt.reshape(-1)
    scales = rng.uniform(0.05, 1.0, N).astype(np.float32)
    dram[0x60000:0x60000 + scales.nbytes] = scales.view(np.uint8)
    commands = [I.ld(0, 0, x.size), I.qact(0, M, 0, kb, kb * D),
                I.tmm(0x40000, 0x60000, 4096, N, kb, wb.shape[1], N, M, 0, 4,
                      unit=unit, rsc=not unit, binary=True),
                I.tmm(0x50000, 0x60000, 8192, N, kb, wt.shape[1], N, M, 0, 4,
                      unit=unit, rsc=not unit), I.halt()]
    return cfg, commands, dram


@pytest.mark.parametrize("kb,unit", [(1, True), (7, False), (8, True), (9, False), (16, False)])
def test_binary_matches_ternary_isa(kb, unit):
    cfg, commands, dram = case(kb, unit)
    sim = Machine(cfg, [commands], [dram]).run().slices[0]
    np.testing.assert_array_equal(sim.tmem[4096:4106], sim.tmem[8192:8202])


@pytest.mark.parametrize("kb,unit", [(1, True), (7, False), (8, True), (9, False), (16, False)])
def test_binary_rtl_matches_isa(have_verilator, kb, unit):
    cfg, commands, dram = case(kb, unit)
    expected = Machine(cfg, [commands], [dram.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [commands], [dram.copy()])
    np.testing.assert_array_equal(tmems[0], expected.slices[0].tmem)
    np.testing.assert_array_equal(drams[0], expected.slices[0].dram)


def test_binary_board_depth_rtl(have_verilator):
    cfg, commands, dram = case(9, False, Config(D=128, MCOLS=8, ACT_BLOCKS=16, TBIN=True))
    expected = Machine(cfg, [commands], [dram.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [commands], [dram.copy()])
    np.testing.assert_array_equal(tmems[0], expected.slices[0].tmem)
    np.testing.assert_array_equal(drams[0], expected.slices[0].dram)


@pytest.mark.parametrize("rows,rowmax", [(2, True), (6, False)])
def test_binary_accumulation_and_replay_rtl(have_verilator, rows, rowmax):
    cfg = Config(D=32, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=16, TBIN=True)
    rng = np.random.default_rng(rows)
    n, kb = 5, 9
    x = rng.standard_normal((rows, kb * cfg.D)).astype(np.float32)
    w = rng.choice(np.array([-1, 1], np.int8), (n, kb * cfg.D))
    wb, wt = pack_binary_mxu(w, cfg.D), pack_ternary_mxu(w, cfg.D)
    initial = rng.standard_normal((rows, n)).astype(np.float32)
    image = np.zeros(cfg.DRAM_BYTES, np.uint8)
    image[:x.nbytes] = x.view(np.uint8).reshape(-1)
    image[0x40000:0x40000 + wb.size] = wb.reshape(-1)
    image[0x50000:0x50000 + wt.size] = wt.reshape(-1)
    image[0x70000:0x70000 + initial.nbytes] = initial.view(np.uint8).reshape(-1)
    commands = [I.ld(0, 0, x.size), I.ld(0x70000, 4096, initial.size),
                I.ld(0x70000, 8192, initial.size), I.qact(0, rows, 0, kb, kb * cfg.D),
                I.tmm(0x40000, 0, 4096, n, kb, wb.shape[1], n, rows, 0, 0,
                      unit=True, binary=True, acc=True, rmax=rowmax),
                I.tmm(0x50000, 0, 8192, n, kb, wt.shape[1], n, rows, 0, 0,
                      unit=True, acc=True, rmax=rowmax), I.halt()]
    sim = Machine(cfg, [commands], [image.copy()]).run().slices[0]
    state = sim.tmem
    length = rows * n + (rows if rowmax else 0)
    np.testing.assert_array_equal(state[4096:4096 + length], state[8192:8192 + length])
    drams, tmems, _ = rtlsim.run(cfg, [commands], [image.copy()])
    np.testing.assert_array_equal(tmems[0], state)
    np.testing.assert_array_equal(drams[0], sim.dram)


def test_binary_requires_capability_and_valid_weights():
    cfg, commands, dram = case(1, True)
    with pytest.raises(SimError, match="Config.TBIN"):
        Machine(Config(D=32, MCOLS=4, ACT_BLOCKS=16), [commands], [dram]).run()
    assert arch_hash(cfg) != arch_hash(Config(D=32, MCOLS=4, ACT_BLOCKS=16))
    with pytest.raises(ValueError):
        pack_binary_mxu(np.array([[0] * 32], dtype=np.int8), 32)
    with pytest.raises(ValueError, match="BIN"):
        I.tmm(0, 0, 0, 1, 1, 32, 1, 1, 0, 4, unit=True, binary=True, quad=True)


def test_compiler_binary_lowering_and_capability():
    cfg = Config(D=32, MCOLS=4, ACT_BLOCKS=16, TBIN=True, PAIR=True, TPAIR=True, TQUAD=True)
    weight = QTensor.binary_weight(0x40000, 0x60000, 5, 9 * cfg.D, cfg.D)
    b = Builder(cfg, 0)
    b.dot(b.alloc((1, 9 * cfg.D)), weight)
    instruction = next(i for i in b.root if i.op == I.TMM)
    assert instruction.w[3] & I.TMM_BIN
    assert not instruction.flags & (I.F_PAIR | I.F_QUAD | I.F_GSC)
    assert weight.rs == 2 * cfg.D
    with pytest.raises(CompileError, match="Config.TBIN"):
        old = Builder(Config(D=32, MCOLS=4, ACT_BLOCKS=16), 0)
        old.dot(old.alloc((1, 9 * cfg.D)), weight)


def test_compiled_binary_dot_on_board_geometry(have_verilator):
    cfg = Config(D=128, MCOLS=8, ACT_BLOCKS=16, TBIN=True, PAIR=True, TPAIR=True, TQUAD=True)
    rng = np.random.default_rng(812)
    k, n = 9 * cfg.D, 5
    x = rng.standard_normal(k).astype(np.float32)
    w = rng.choice(np.array([-1, 1], np.int8), (n, k))
    wb = pack_binary_mxu(w, cfg.D)
    wt = pack_ternary_mxu(w, cfg.D)
    image = np.zeros(cfg.DRAM_BYTES, np.uint8)
    image[:x.nbytes] = x.view(np.uint8)
    image[0x40000:0x40000 + wb.size] = wb.reshape(-1)
    image[0x50000:0x50000 + wt.size] = wt.reshape(-1)
    b = Builder(cfg, 0)
    a = b.alloc((1, k))
    b.emit(I.ld(0, a.base, k))
    yb = b.dot(a, QTensor.binary_weight(0x40000, None, n, k, cfg.D))
    yt = b.dot(a, QTensor.ternary_weight(0x50000, None, n, k, cfg.D))
    b.emit(I.st(0x10000, yb.base, n))
    b.emit(I.st(0x11000, yt.base, n))
    commands = b.finish()
    expected = Machine(cfg, [commands], [image.copy()]).run().slices[0]
    np.testing.assert_array_equal(expected.dram[0x10000:0x10000 + 4 * n],
                                  expected.dram[0x11000:0x11000 + 4 * n])
    drams, tmems, _ = rtlsim.run(cfg, [commands], [image.copy()])
    np.testing.assert_array_equal(drams[0], expected.dram)
    np.testing.assert_array_equal(tmems[0], expected.tmem)


def test_host_rejects_new_opcodes_on_old_card():
    from opentpu.host.board import Board, ConfigMismatch
    from opentpu.host.fake import FakeTransport
    board = Board(FakeTransport(devname=None, regmap=5))
    cfg, commands, _ = case(1, True)
    with pytest.raises(ConfigMismatch, match="TBIN"):
        board.load_program(4096, I.assemble(commands))
    with pytest.raises(ConfigMismatch, match="VCMP"):
        board.load_program(4096, I.assemble([I.vop(I.V_CMP_EQ, 1, 2, 3, 1, 8, 8, 8, 8), I.halt()]))
