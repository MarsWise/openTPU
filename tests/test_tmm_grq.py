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


@pytest.mark.parametrize("parallel_flag", [I.F_PAIR, I.F_QUAD])
@pytest.mark.parametrize("grq", [False, True])
def test_gsc_rejects_pair_quad_until_separate_block_scales_are_supported(parallel_flag, grq):
    cfg, image, head, gsc, grq_op, _ = case(2, 1, False, 4)
    gq4 = grq and parallel_flag == I.F_QUAD      # GRQ QUAD: its own capability (Config.TGQ4)
    if not gq4:
        with pytest.raises(ValueError, match="GSC"):
            I.tmm(W_GRQ if grq else W_ROW, S_GRQ if grq else S_ROW, 8192, 4, 2, 128, 7, 1, 0,
                  16 if grq else 8, gsc=True, grq=grq,
                  pair=parallel_flag == I.F_PAIR, quad=parallel_flag == I.F_QUAD)
    base = grq_op if grq else gsc
    raw = dataclasses.replace(base, flags=base.flags | parallel_flag)
    with pytest.raises(SimError, match="TGQ4" if gq4 else "GSC"):
        Machine(cfg, [head + [raw, I.halt()]], [image.copy()]).run()


def test_gsc_requires_per_block_scaling_before_accumulation():
    from opentpu import fp32 as F
    cfg, image, head, gsc, _, (x, w, _, _, out, ors, *_) = case(2, 1, False, 4)
    scales = np.tile(np.array([.25, 1.5], np.float32), (4, 1))
    image[S_ROW:S_ROW + scales.nbytes] = scales.view(np.uint8).reshape(-1)
    result = Machine(cfg, [head + [gsc, I.halt()]], [image]).run().slices[0]
    actual = result.tmem[out:out + ors].view(np.float32)[:4]
    q, row_scale, offsets = F.quantize_mx(x, 32)
    dequant = (q.reshape(1, 8, 32).astype(np.float32) *
               np.exp2(-offsets.astype(np.float32))[:, :, None] * row_scale[:, None, None]).reshape(1, 256)
    a = dequant[:, :128] @ w[:, :128].T
    b = dequant[:, 128:] @ w[:, 128:].T
    expected = (a * .25 + b * 1.5).reshape(-1)
    wrong_merged = ((a + b) * .25).reshape(-1)
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 5e-6
    assert np.linalg.norm(actual - wrong_merged) / np.linalg.norm(expected) > .25


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


@pytest.mark.parametrize("n,kb,m,acc", [(4, 8, 1, False), (28, 9, 7, True), (36, 15, 8, False)])
def test_grq_full_precision_group_scales_and_tails(n, kb, m, acc):
    cfg, img, head, gsc, grq, (*_, srs) = case(kb, m, acc, n, mcols=8)
    rng = np.random.default_rng(8000 + n)
    # Deliberately not FP16 values: distinct scales per row and K group, with zeros.
    scales = rng.uniform(1e-4, 2, (n, kb)).astype(np.float32)
    scales[0, ::2] = 0
    img[S_ROW:S_ROW + scales.nbytes] = scales.view(np.uint8).reshape(-1)
    planes, stride = grq_scales(scales)
    assert stride == srs
    for k in range(kb):
        img[S_GRQ + k * srs:S_GRQ + (k + 1) * srs] = planes[k].view(np.uint8)
    a = Machine(cfg, [head + [gsc, I.halt()]], [img.copy()]).run().slices[0]
    b = Machine(cfg, [head + [grq, I.halt()]], [img.copy()]).run().slices[0]
    np.testing.assert_array_equal(a.tmem, b.tmem)


@pytest.mark.parametrize("waitc", [False, True])
def test_grq_weight_write_hazard_with_long_axi_stalls_rtl(have_verilator, waitc):
    from opentpu import rtlsim
    cfg, img, head, gsc, grq, (*_, rs, srs) = case(5, 2, True, 20)
    cfg = dataclasses.replace(cfg, SEQ_WC=waitc)
    # A DMA write overlaps a later GRQ weight plane; preserve the packed bytes
    # so the test also exercises ordering without inventing reserved symbols.
    prog = head + [I.ld(W_GRQ + rs, 20000, 32), I.st(W_GRQ + rs, 20000, 32),
                   grq, I.st(S_GRQ + 4 * srs, 10000, 12), gsc, I.halt()]
    ref = Machine(cfg, [prog], [img.copy()]).run().slices[0]
    dram, tmem, _ = rtlsim.run(cfg, [prog], [img.copy()], axi=True, stall=70, seed=991)
    np.testing.assert_array_equal(dram[0], ref.dram)
    np.testing.assert_array_equal(tmem[0], ref.tmem)


# ---- GRQ QUAD (Config.TGQ4): a GRQ chunk's four quarters in one advance, M 1 (C2)
def gq4_case(kb, acc, n, mcols=4, seed=0, m=1, ors=None):
    cfg, image, _, gsc, grq, rest = case(kb, m, acc, n, rmax=False, seed=seed, mcols=mcols)
    cfg = dataclasses.replace(cfg, TQUAD=True, TPAIR=True, PAIR=True, TGQ4=True, TGQ4M=m > 1)
    x, _, _, initial, out, ors0, rs, srs = rest
    if ors is not None:                     # (another output row stride: the drain's lanes)
        ors0 = ors
        grq = I.tmm(W_GRQ, S_GRQ, out, n, kb, rs, ors, m, 0, srs, gsc=True, grq=True, acc=acc)
    head = [I.ld(0, 0, x.size), I.qact(0, m, 0, kb, kb * cfg.D, dup=True, dup4=True)]
    if acc:
        head.append(I.ld(INIT, out, initial.size))
    quad = I.tmm(W_GRQ, S_GRQ, out, n, kb, rs, ors0, m, 0, srs, gsc=True, grq=True, quad=True,
                 acc=acc)
    return cfg, image, head, grq, quad


GQ4 = [(1, False, 4), (4, True, 8), (5, False, 20), (7, True, 36), (15, False, 12)]


@pytest.mark.parametrize("kb,acc,n", GQ4)
def test_gq4_isa_is_grq_bit_for_bit(kb, acc, n):
    cfg, image, head, grq, quad = gq4_case(kb, acc, n)
    a = Machine(cfg, [head + [grq, I.halt()]], [image.copy()]).run().slices[0]
    b = Machine(cfg, [head + [quad, I.halt()]], [image.copy()]).run().slices[0]
    np.testing.assert_array_equal(a.tmem, b.tmem)


def test_gq4_refusals():
    cfg, image, head, grq, quad = gq4_case(3, False, 8)
    with pytest.raises(ValueError, match="GRQ QUAD"):       # M 3
        I.tmm(W_GRQ, S_GRQ, 8192, 8, 3, 512, 11, 3, 0, 32, gsc=True, grq=True, quad=True)
    with pytest.raises(ValueError, match="GRQ QUAD"):       # RMAX
        I.tmm(W_GRQ, S_GRQ, 8192, 8, 3, 512, 11, 1, 0, 32, gsc=True, grq=True, quad=True,
              rmax=True)
    with pytest.raises(SimError, match="TGQ4"):             # the bitstream's capability
        Machine(dataclasses.replace(cfg, TGQ4=False), [head + [quad, I.halt()]],
                [image.copy()]).run()
    # the row's copies in ACT rows 1..3 are what the columns read: without QACT DUP4 the
    # simulator computes from them too (zeros here), not from row 0
    plain = [head[0], I.qact(0, 1, 0, 3, 3 * cfg.D)] + head[2:]
    a = Machine(cfg, [head + [quad, I.halt()]], [image.copy()]).run().slices[0]
    try:
        b = Machine(cfg, [plain + [quad, I.halt()]], [image.copy()]).run().slices[0]
    except SimError:
        return                                               # (different Sa: refused)
    assert not np.array_equal(a.tmem, b.tmem)


@pytest.mark.parametrize("kb,acc,n", GQ4)
@pytest.mark.parametrize("mcols", [4, 8])
def test_gq4_rtl(have_verilator, kb, acc, n, mcols):
    """RTL == ISA (AXI stalls): GRQ QUAD alone, then between a GRQ and a serial GSC command of
    the same operand (the queue's two entries, the drain's lane counts changing)."""
    from opentpu import rtlsim
    cfg, image, head, grq, quad = gq4_case(kb, acc, n, mcols=mcols)
    prog = head + [quad, grq, quad, I.halt()]
    ref = Machine(cfg, [prog], [image.copy()]).run().slices[0]
    drams, tmems, _ = rtlsim.run(cfg, [prog], [image.copy()], axi=True, stall=35, seed=kb + n)
    np.testing.assert_array_equal(tmems[0], ref.tmem)
    np.testing.assert_array_equal(drams[0], ref.dram)


# ---- GRQ QUAD M 2 (Config.TGQ4M, CAPS2 bit21): column j row j % 2, quarter j // 2
GQ4M = [(1, False, 4, None), (4, True, 8, 12), (5, False, 20, 23), (7, True, 36, 44),
        (3, True, 16, 16)]


@pytest.mark.parametrize("kb,acc,n,ors", GQ4M)
def test_gq4m_isa_is_grq_bit_for_bit(kb, acc, n, ors):
    cfg, image, head, grq, quad = gq4_case(kb, acc, n, mcols=8, m=2, ors=ors)
    a = Machine(cfg, [head + [grq, I.halt()]], [image.copy()]).run().slices[0]
    b = Machine(cfg, [head + [quad, I.halt()]], [image.copy()]).run().slices[0]
    np.testing.assert_array_equal(a.tmem, b.tmem)


def test_gq4m_refusals():
    cfg, image, head, grq, quad = gq4_case(3, False, 8, mcols=8, m=2)
    with pytest.raises(SimError, match="TGQ4M"):            # M 2's own capability
        Machine(dataclasses.replace(cfg, TGQ4M=False), [head + [quad, I.halt()]],
                [image.copy()]).run()
    cfg4, image4, head4, _, _ = gq4_case(3, False, 8, mcols=4, m=1)
    quad4 = I.tmm(W_GRQ, S_GRQ, 8192, 8, 3, 512, 11, 2, 0, 32, gsc=True, grq=True, quad=True)
    head4 = [head4[0], I.qact(0, 2, 0, 3, 3 * cfg4.D)]
    with pytest.raises(SimError, match="GRQ QUAD"):         # 4 M <= MCOLS
        Machine(dataclasses.replace(cfg4, TGQ4M=True), [head4 + [quad4, I.halt()]],
                [image4.copy()]).run()


@pytest.mark.parametrize("kb,acc,n,ors", GQ4M)
def test_gq4m_rtl(have_verilator, kb, acc, n, ors):
    """RTL == ISA (AXI stalls), MCOLS 8, M 2: GRQ QUAD alone, then between a GRQ and itself;
    row strides whose drain takes 8, 4, 2 or 1 lanes a cycle (ors % 8: 4, 7 / 3, 0 ...)."""
    from opentpu import rtlsim
    cfg, image, head, grq, quad = gq4_case(kb, acc, n, mcols=8, m=2, ors=ors)
    prog = head + [quad, grq, quad, I.halt()]
    ref = Machine(cfg, [prog], [image.copy()]).run().slices[0]
    drams, tmems, _ = rtlsim.run(cfg, [prog], [image.copy()], axi=True, stall=35, seed=kb + n)
    np.testing.assert_array_equal(tmems[0], ref.tmem)
    np.testing.assert_array_equal(drams[0], ref.dram)


def test_gq4m_drain_lanes_and_cycles_rtl(have_verilator):
    """M 2 at ors = 4 mod 8 drains 8 columns a cycle (MXU_E 8, the m8 card's): its GRQ QUAD
    takes about the cycles of M 1's (and far fewer than plain GRQ's four replay groups)."""
    from opentpu import rtlsim
    cyc = {}
    for name, m, ors in (("m1", 1, None), ("m2", 2, 260), ("grq2", 2, 260)):
        cfg, image, head, grq, quad = gq4_case(4, True, 256, mcols=8, m=m, ors=ors)
        prog = head + [grq if name == "grq2" else quad, I.halt()]
        ref = Machine(cfg, [prog], [image.copy()]).run().slices[0]
        drams, tmems, st = rtlsim.run(cfg, [prog], [image.copy()], uarch={"MXU_E": 8})
        np.testing.assert_array_equal(tmems[0], ref.tmem)
        c = st["cycles"]
        cyc[name] = c[0] if isinstance(c, (list, tuple)) else c
    assert cyc["m2"] < 1.25 * cyc["m1"], cyc
    assert cyc["m2"] < 0.6 * cyc["grq2"], cyc


def test_gq4_firmware_capability_is_required_before_program_load():
    from opentpu.host.board import Board, ConfigMismatch, device_config
    from opentpu.host.fake import FakeTransport
    prog = I.assemble([I.tmm(0, 0, 0, 8, 4, 256, 8, 1, 0, 32, gsc=True, grq=True, quad=True),
                       I.halt()])
    with pytest.raises(ConfigMismatch, match="TGQ4"):
        Board(FakeTransport(regmap=5, devname=None, tgsc=True, tgrq=True)).load_program(0x1000, prog)
    board = Board(FakeTransport(regmap=5, devname=None, tgsc=True, tgrq=True, tgq4=True))
    assert device_config(board.info()).TGQ4
    board.load_program(0x1000, prog)



@pytest.mark.parametrize("n,kb", [(256, 6), (64, 9)])
def test_gsc_groups_pipelined_rtl(have_verilator, n, kb):
    """otpu_tmm_gsc PIPE (the default): a K block's subcommand is handed to the MXU while the one
    before streams and drains. The results are the serial adapter's (PIPE 0) and the ISA's bit for
    bit (GRQ QUAD, then GRQ and a serial GSC command on the same output with ACC), with AXI stalls,
    and the command takes fewer cycles."""
    from opentpu import rtlsim
    cfg, image, head, grq, quad = gq4_case(kb, True, n, mcols=8)
    _, _, _, gsc, _, _ = case(kb, 1, True, n, rmax=False, mcols=8)
    cfg = dataclasses.replace(cfg, TQUAD=True, TPAIR=True, PAIR=True, TGQ4=True, DRAM_BYTES=1 << 20)
    prog = head + [quad, grq, gsc, I.halt()]
    ref = Machine(cfg, [prog], [image.copy()]).run().slices[0]
    cyc = {}
    for pipe in (0, 1):
        drams, tmems, st = rtlsim.run(cfg, [prog], [image.copy()], axi=True, stall=30, seed=n + kb,
                                      uarch={"GSC_PIPE": pipe})
        np.testing.assert_array_equal(tmems[0], ref.tmem)
        np.testing.assert_array_equal(drams[0], ref.dram)
        c = st["cycles"]
        cyc[pipe] = c[0] if isinstance(c, list) else c
    assert cyc[1] < cyc[0], cyc


def test_gq4m_firmware_capability_is_required_before_program_load():
    """A program with a 2-row GRQ QUAD loads only on firmware with CAPS2 bit21 (TGQ4M); the
    device's Config then has TGQ4M (and its CFG_HASH the bit)."""
    from opentpu.host.board import Board, ConfigMismatch, device_config
    from opentpu.host.fake import FakeTransport
    prog = I.assemble([I.tmm(0, 0, 0, 8, 4, 256, 12, 2, 0, 32, gsc=True, grq=True, quad=True),
                       I.halt()])
    one = I.assemble([I.tmm(0, 0, 0, 8, 4, 256, 12, 1, 0, 32, gsc=True, grq=True, quad=True),
                      I.halt()])
    old = Board(FakeTransport(regmap=5, devname=None, tgsc=True, tgrq=True, tgq4=True))
    old.load_program(0x1000, one)                       # (M 1 runs on TGQ4 alone)
    with pytest.raises(ConfigMismatch, match="TGQ4M"):
        old.load_program(0x1000, prog)
    board = Board(FakeTransport(regmap=5, devname=None, tgsc=True, tgrq=True, tgq4=True,
                                tgq4m=True))
    assert device_config(board.info()).TGQ4M
    board.load_program(0x1000, prog)
