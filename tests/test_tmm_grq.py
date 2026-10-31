"""TMM GSC GRQ: the block-major layout streams every packed chunk once with GSC's exact math."""
import dataclasses

import numpy as np
import pytest

from opentpu import isa as I
from opentpu.isasim import Config, Machine, SimError
from opentpu.quant import grq_scales, pack_ternary_grq, pack_ternary_mxu

W_ROW, S_ROW, W_GRQ, S_GRQ, INIT = 0x20000, 0x40000, 0x60000, 0x80000, 0x50000


def case(kb=5, m=3, acc=False, n=12, rmax=True, seed=0, mcols=4):
    cfg = Config(S=1, D=128, MCOLS=mcols, ACT_ROWS=2 * mcols, ACT_BLOCKS=16, TGSC=True,
                 TGRQ=True)
    D = cfg.D
    rng = np.random.default_rng(2600 + 31 * kb + 7 * m + n + seed)
    ors, out = n + 3, 8192
    x = rng.normal(size=(m, kb * D)).astype(np.float32)
    w = rng.integers(-1, 2, (n, kb * D), dtype=np.int8)
    scales = rng.uniform(.001, .2, (n, kb)).astype(np.float16).astype(np.float32)
    rows = pack_ternary_mxu(w, D)
    planes = pack_ternary_grq(w, D)
    splanes, srs = grq_scales(scales)
    rs = planes.shape[1] + 2 * D                 # planes apart by more than their size
    image = np.zeros(cfg.DRAM_BYTES, np.uint8)
    image[:x.nbytes] = x.view(np.uint8).reshape(-1)
    image[W_ROW:W_ROW + rows.size] = rows.reshape(-1)
    image[S_ROW:S_ROW + scales.nbytes] = scales.view(np.uint8).reshape(-1)
    for k in range(kb):
        image[W_GRQ + k * rs:W_GRQ + k * rs + planes.shape[1]] = planes[k]
        image[S_GRQ + k * srs:S_GRQ + (k + 1) * srs] = splanes[k].view(np.uint8)
    initial = rng.normal(size=(m, ors)).astype(np.float32)
    image[INIT:INIT + initial.nbytes] = initial.view(np.uint8).reshape(-1)
    head = [I.ld(0, 0, x.size), I.qact(0, m, 0, kb, kb * D)]
    if acc:
        head.append(I.ld(INIT, out, initial.size))
    rmax = rmax and m <= cfg.MCOLS
    gsc = I.tmm(W_ROW, S_ROW, out, n, kb, rows.shape[1], ors, m, 0, 4 * kb,
                gsc=True, acc=acc, rmax=rmax)
    grq = I.tmm(W_GRQ, S_GRQ, out, n, kb, rs, ors, m, 0, srs, gsc=True, grq=True,
                acc=acc, rmax=rmax)
    return cfg, image, head, gsc, grq, (x, w, scales, initial, out, ors, rs, srs)


CASES = [(1, 1, False, 4), (3, 4, True, 12), (4, 1, True, 8), (5, 2, False, 20),
         (7, 3, False, 16), (2, 4, True, 4)]


@pytest.mark.parametrize("kb,m,acc,n", CASES)
def test_grq_isa_is_gsc_bit_for_bit(kb, m, acc, n):
    cfg, image, head, gsc, grq, (x, w, s, initial, out, ors, *_) = case(kb, m, acc, n)
    a = Machine(cfg, [head + [gsc, I.halt()]], [image.copy()]).run().slices[0]
    b = Machine(cfg, [head + [grq, I.halt()]], [image.copy()]).run().slices[0]
    np.testing.assert_array_equal(a.tmem, b.tmem)
    actual = b.tmem[out:out + m * ors].view(np.float32).reshape(m, ors)[:, :n]
    expected = sum((x[:, k*128:(k+1)*128] @ w[:, k*128:(k+1)*128].T) * s[:, k]
                   for k in range(kb)) + (initial[:, :n] if acc else 0)
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < .012


def test_grq_layout_packers():
    w = np.arange(8 * 256).reshape(8, 256) % 3 - 1
    p = pack_ternary_grq(w, 128)
    assert p.shape == (2, 2 * 128)
    rows = pack_ternary_mxu(w, 128)
    # plane 1, chunk 1, quarter 2: row 6's block 1, row-major chunk 0 quarter 1 of row 6
    np.testing.assert_array_equal(p[1, 128 + 64:128 + 96], rows[6, 32:64])
    sp, srs = grq_scales(np.arange(16, dtype=np.float32).reshape(8, 2))
    assert srs == 32 and sp.shape == (2, 8) and sp[1, 3] == 7
    with pytest.raises(ValueError, match="N % 4"):
        pack_ternary_grq(w[:6], 128)


def test_grq_refuses_bad_configuration():
    cfg, image, head, gsc, grq, (*_, rs, srs) = case(3, 2, False, 8)
    with pytest.raises(SimError, match="TGRQ"):
        Machine(dataclasses.replace(cfg, TGRQ=False), [head + [grq, I.halt()]], [image]).run()
    with pytest.raises(ValueError, match="GRQ"):
        I.tmm(W_GRQ, S_GRQ, 0, 6, 3, rs, 8, 1, 0, srs, gsc=True, grq=True)      # N % 4
    with pytest.raises(ValueError, match="GRQ"):
        I.tmm(W_GRQ, S_GRQ + 4, 0, 8, 3, rs, 8, 1, 0, srs, gsc=True, grq=True)  # ssa % 16
    with pytest.raises(ValueError, match="GRQ"):
        I.tmm(W_GRQ, S_GRQ, 0, 8, 3, rs, 8, 1, 0, 16, gsc=True, grq=True)       # srs < 4N
    with pytest.raises(ValueError, match="GRQ"):
        I.tmm(W_GRQ, S_GRQ, 0, 8, 3, rs, 8, 1, 0, srs, grq=True, rsc=True)
    bad = I.tmm(W_GRQ, S_GRQ, 0, 8, 3, rs, 8, 5, 0, srs, gsc=True, grq=True)   # M > MCOLS
    with pytest.raises(SimError, match="GRQ"):
        Machine(cfg, [[I.ld(0, 0, 5 * 384), I.qact(0, 5, 0, 3, 384), bad, I.halt()]],
                [image]).run()


def test_grq_firmware_capability_is_required_before_program_load():
    from opentpu.host.board import Board, ConfigMismatch, device_config
    from opentpu.host.fake import FakeTransport
    prog = I.assemble([I.tmm(0, 0, 0, 8, 4, 256, 8, 1, 0, 32, gsc=True, grq=True), I.halt()])
    with pytest.raises(ConfigMismatch, match="TGRQ"):
        Board(FakeTransport(regmap=5, devname=None, tgsc=True)).load_program(0x1000, prog)
    board = Board(FakeTransport(regmap=5, devname=None, tgsc=True, tgrq=True))
    assert device_config(board.info()).TGRQ
    board.load_program(0x1000, prog)


@pytest.mark.parametrize("kb,m,acc,n", CASES)
def test_grq_rtl(have_verilator, kb, m, acc, n):
    from opentpu import rtlsim
    cfg, image, head, gsc, grq, _ = case(kb, m, acc, n)
    prog = head + [grq, I.halt()]
    ref = Machine(cfg, [prog], [image.copy()]).run().slices[0]
    drams, tmems, _ = rtlsim.run(cfg, [prog], [image.copy()], axi=True, stall=35, seed=kb + n)
    np.testing.assert_array_equal(tmems[0], ref.tmem)
    np.testing.assert_array_equal(drams[0], ref.dram)


@pytest.mark.parametrize("waitc", [False, True])
def test_grq_gsc_and_legacy_share_completions_and_hazards_rtl(have_verilator, waitc):
    """GRQ between legacy RSC and serial GSC commands, then a store over its last scale plane
    and a load over its output: the footprints (otpu_pkg, waits.py) must hold them back."""
    from opentpu import rtlsim
    cfg, image, head, gsc, grq, (*_, out, ors, rs, srs) = case(5, 1, False, 12)
    cfg = dataclasses.replace(cfg, SEQ_WC=waitc)
    legacy = I.tmm(W_ROW, S_ROW, 10000, 12, 5, 256, 13, 1, 0, 20, rsc=True)
    prog = head + [legacy, grq, legacy, gsc, grq,
                   I.st(S_GRQ + 4 * srs, 10000, 12),        # over plane 4's scales
                   I.ld(INIT, out, 13), I.halt()]
    ref = Machine(cfg, [prog], [image.copy()]).run().slices[0]
    drams, tmems, _ = rtlsim.run(cfg, [prog], [image.copy()], axi=True, stall=45, seed=71)
    np.testing.assert_array_equal(tmems[0], ref.tmem)
    np.testing.assert_array_equal(drams[0], ref.dram)


def test_compiled_hadamard_grq_linear_matches_row_major_and_rtl(have_verilator):
    """The compiler's GRQ descriptor (row and K slices, Hadamard signs after the scale planes)
    against the row-major GSC one on the same matrix; then the RTL against the ISA."""
    from opentpu import language as ol, rtlsim
    from opentpu import quant as Q
    from opentpu.compiler import QTensor
    from opentpu.runtime import Input, Output, compile_kernel, run
    rng = np.random.default_rng(912)
    n, k = 12, 1024
    groups = rng.uniform(.01, .1, (n, k // 128)).astype(np.float16).astype(np.float32)
    W = (rng.integers(-1, 2, (n, k)).reshape(n, -1, 128) * groups[..., None]).reshape(n, k)
    signs = rng.choice(np.array([-1, 1], np.float32), k)
    x = rng.normal(size=(3, k)).astype(np.float32)
    cfg = Config(D=128, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=16, TGSC=True, TGRQ=True)
    outs = {}
    for grq in (False, True):
        w = QTensor.weight(0x40000, 0x60000, n, k, 'ternary_gsc', 128, grq=grq)

        @ol.jit
        def linear(src, dst, w):
            ol.store(dst[:, 0:8], ol.dot(ol.load(src), w[0:8, :]))
            ol.store(dst[:, 8:12], ol.dot(ol.load(src), w[8:12, :]))

        compiled, images = compile_kernel(linear, cfg, src=Input(x), dst=Output((3, n)), w=w)
        packed, scales = (Q.pack_ternary_gsc_grq if grq else Q.pack_ternary_gsc)(W, 128)
        for addr, data in ((0x40000, packed), (0x60000, scales),
                           (0x60000 + scales.nbytes, signs)):
            raw = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
            images[0][addr:addr + len(raw)] = raw
        tm = [i for p in compiled.programs for i in p if i.op == I.TMM]
        assert tm and all(bool(i.w[3] & I.TMM_GRQ) == grq for i in tm)
        ref = run(compiled, [images[0].copy()], 'isa')
        outs[grq] = ref.outputs['dst']
    np.testing.assert_array_equal(outs[True], outs[False])
    drams, tmems, _ = rtlsim.run(cfg, compiled.programs, images, axi=True, stall=35, seed=892)
    np.testing.assert_array_equal(drams[0], ref.drams[0])
    np.testing.assert_array_equal(tmems[0], ref.tmems[0])


@pytest.mark.parametrize("m,acc", [(8, True), (5, False)])
def test_grq_rtl_m8(have_verilator, m, acc):
    """The card's MCOLS 8 build: up to eight stationary rows, every group draining M results."""
    from opentpu import rtlsim
    cfg, image, head, gsc, grq, _ = case(6, m, acc, 16, mcols=8)
    prog = head + [grq, gsc, I.halt()]
    ref = Machine(cfg, [prog], [image.copy()]).run().slices[0]
    drams, tmems, _ = rtlsim.run(cfg, [prog], [image.copy()], axi=True, stall=35, seed=m)
    np.testing.assert_array_equal(tmems[0], ref.tmem)
    np.testing.assert_array_equal(drams[0], ref.dram)
