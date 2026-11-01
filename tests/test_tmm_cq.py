"""TMM CQ (Config.TCQ, opentpu/cq.py): codebook-quantized weights (Cactus CQ1/2/3/4) streamed as
packed codes, decoded on the card through the LUTSET LUT, with GSC's fp32 per-group scales."""
import dataclasses

import numpy as np
import pytest

from opentpu import cq as CQ
from opentpu import isa as I
from opentpu.isasim import Config, Machine, SimError

W_ROW, S_ROW, INIT, OUT = 0x20000, 0x60000, 0x50000, 8192


def case(bits=4, kb=3, m=2, n=12, acc=False, rmax=True, seed=0, cb=None, waitc=False):
    cfg = Config(S=1, D=128, MCOLS=8, ACT_ROWS=16, ACT_BLOCKS=16, TGSC=True, TBIN=True,
                 TBGSC=True, TCQ=True, SEQ_WC=waitc)
    D = cfg.D
    rng = np.random.default_rng(900 + 17 * bits + 5 * kb + m + n + seed)
    ors = n + 3
    nc = 1 << min(bits, 4) if bits != 3 else 8
    codebook = rng.normal(size=nc).astype(np.float16).astype(np.float32) if cb is None else cb
    codes = rng.integers(0, nc, (n, kb * D), dtype=np.uint8)
    norms = rng.uniform(.01, .3, (n, kb)).astype(np.float32)
    lut, s = CQ.lut(codebook)
    rows = CQ.pack(codes, bits, D)
    sw = CQ.scales(norms, s)
    x = rng.normal(size=(m, kb * D)).astype(np.float32)
    image = np.zeros(cfg.DRAM_BYTES, np.uint8)
    image[:x.nbytes] = x.view(np.uint8).reshape(-1)
    image[W_ROW:W_ROW + rows.size] = rows.reshape(-1)
    image[S_ROW:S_ROW + sw.nbytes] = sw.view(np.uint8).reshape(-1)
    initial = rng.normal(size=(m, ors)).astype(np.float32)
    image[INIT:INIT + initial.nbytes] = initial.view(np.uint8).reshape(-1)
    prog = [I.ld(0, 0, x.size), I.qact(0, m, 0, kb, kb * D)]
    if acc:
        prog.append(I.ld(INIT, OUT, initial.size))
    prog += I.tmm_lut(lut)
    prog.append(I.tmm_cq(W_ROW, S_ROW, OUT, n, kb, rows.shape[1], ors, m, 0, 4 * kb, bits,
                         acc=acc, rmax=rmax and m <= cfg.MCOLS))
    prog.append(I.halt())
    return cfg, image, prog, (x, codes, norms, codebook, initial, ors)


CASES = [(1, 3, 2, 8), (2, 5, 1, 12), (3, 2, 3, 4), (4, 3, 2, 12), (4, 1, 8, 16), (2, 1, 1, 4)]


@pytest.mark.parametrize("bits,kb,m,n", CASES)
@pytest.mark.parametrize("acc", [False, True])
def test_cq_isa_matches_the_codebook_matrix(bits, kb, m, n, acc):
    cfg, image, prog, (x, codes, norms, cb, initial, ors) = case(bits, kb, m, n, acc)
    s = Machine(cfg, [prog], [image]).run().slices[0]
    got = s.tmem[OUT:OUT + m * ors].view(np.float32).reshape(m, ors)[:, :n]
    want = x @ CQ.dense(codes, norms, cb).T + (initial[:, :n] if acc else 0)
    dev = x @ CQ.device_dense(codes, norms, cb).T + (initial[:, :n] if acc else 0)
    # the activations' int8 (QACT) and the LUT's int8: ~1 % against the float codebook
    assert np.linalg.norm(got - dev) / np.linalg.norm(dev) < .012
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < .02


def test_cq_packers_and_lut():
    cb = np.array([-1.5, -.5, .25, 2.0], np.float32)
    q, s = CQ.lut(cb)
    assert q[3] == 127 and np.abs(q * s - cb).max() <= s / 2 + 1e-7
    codes = (np.arange(2 * 256) % 16).reshape(2, 256).astype(np.uint8)
    p4 = CQ.pack(codes, 4)
    assert p4.shape == (2, 128) and p4[0, 0] == (0 | (1 << 4)) and p4[0, 64] == codes[0, 128] | (codes[0, 129] << 4)
    p2 = CQ.pack(codes % 4, 2)
    assert p2.shape == (2, 128) and p2[0, 0] == 0 | (1 << 2) | (2 << 4) | (3 << 6)
    p1 = CQ.pack(codes % 2, 1)
    assert p1.shape == (2, 128) and p1[0, 0] == 0b10101010
    with pytest.raises(ValueError):
        CQ.pack(codes, 2)                    # codes past 2 bits


def test_lutset_holds_until_the_next_one():
    """Two CQ TMMs with different LUTs back to back: each decodes with its own (LUTSET is in
    order), the second LUT replacing the first only after the first TMM."""
    cfg, image, prog, (x, codes, norms, cb, *_ ) = case(2, 2, 1, 8, rmax=False)
    lut2, _ = CQ.lut(cb[::-1].copy())
    second = I.tmm_cq(W_ROW, S_ROW, OUT + 64, 8, 2, CQ.pack(codes, 2).shape[1], 8, 1, 0, 8, 2)
    p = prog[:-1] + I.tmm_lut(lut2) + [second, I.halt()]
    t = Machine(cfg, [p], [image]).run().slices[0].tmem
    a = t[OUT:OUT + 8].view(np.float32)
    b = t[OUT + 64:OUT + 72].view(np.float32)
    lut1, s1 = CQ.lut(cb)
    want_b = x[:1] @ ((lut2.astype(np.float32)[codes].reshape(8, 2, 128) *
                       CQ.scales(norms, s1)[:, :, None]).reshape(8, 256)).T
    assert not np.allclose(a, b)
    assert np.linalg.norm(b - want_b[0]) / np.linalg.norm(want_b) < .012


def test_cq_refusals():
    cfg, image, prog, _ = case(4, 2, 1, 4)
    with pytest.raises(SimError, match="TCQ"):
        Machine(dataclasses.replace(cfg, TCQ=False), [prog], [image]).run()
    with pytest.raises(ValueError):
        I.tmm_cq(W_ROW, S_ROW, 0, 4, 2, 256, 4, 1, 0, 8, 5)
    with pytest.raises(ValueError):
        I.tmm_lut([200])
    from opentpu.host.board import Board, ConfigMismatch
    from opentpu.host.fake import FakeTransport
    code = I.assemble(prog)
    with pytest.raises(ConfigMismatch, match="TCQ"):
        Board(FakeTransport(devname=None, regmap=5, tgsc=True, tbin=True, tbgsc=True)
              ).load_program(0x1000, code)
    Board(FakeTransport(devname=None, regmap=5, tgsc=True, tbin=True, tbgsc=True, tcq=True)
          ).load_program(0x1000, code)


@pytest.mark.parametrize("bits,kb,m,n,acc,waitc", [(1, 3, 2, 8, False, False),
                                                  (2, 5, 1, 12, True, True),
                                                  (3, 2, 3, 4, False, True),
                                                  (4, 3, 8, 12, True, False)])
def test_cq_rtl_matches_isa(have_verilator, bits, kb, m, n, acc, waitc):
    from opentpu import rtlsim
    cfg, image, prog, _ = case(bits, kb, m, n, acc, waitc=waitc)
    ref = Machine(cfg, [prog], [image.copy()]).run().slices[0]
    drams, tmems, _ = rtlsim.run(cfg, [prog], [image.copy()], axi=True, stall=35,
                                 seed=bits + kb + n)
    np.testing.assert_array_equal(tmems[0], ref.tmem)
    np.testing.assert_array_equal(drams[0], ref.dram)


def test_lutset_between_cq_tmms_rtl(have_verilator):
    from opentpu import rtlsim
    cfg, image, prog, (x, codes, norms, cb, *_ ) = case(2, 2, 1, 8, rmax=False)
    lut2, _ = CQ.lut(cb[::-1].copy())
    second = I.tmm_cq(W_ROW, S_ROW, OUT + 64, 8, 2, CQ.pack(codes, 2).shape[1], 8, 1, 0, 8, 2)
    p = prog[:-1] + I.tmm_lut(lut2) + [second, I.halt()]
    ref = Machine(cfg, [p], [image.copy()]).run().slices[0]
    drams, tmems, _ = rtlsim.run(cfg, [p], [image.copy()], axi=True, stall=35, seed=12)
    np.testing.assert_array_equal(tmems[0], ref.tmem)


def test_compiled_cq_linear():
    """ol.dot over a QTensor.cq_weight: the compiler emits the LUTSETs and the TMM CQs, K split
    into QACT ranges at chunk boundaries; against the float codebook matrix."""
    from opentpu import language as ol
    from opentpu.compiler import QTensor, Tensor
    from opentpu.runtime import Input, Output, launch, Weight  # noqa: F401

    for bits in (2, 4):
        cfg, image, prog, (x, codes, norms, cb, *_ ) = case(bits, 4, 3, 16, rmax=False)
        lut, s = CQ.lut(cb)
        rows = CQ.pack(codes, bits)

        @ol.jit
        def lin(xin, y, M: int, N: int, K: int):
            w = QTensor.cq_weight(W_ROW, S_ROW, N, K, 128, bits, lut, rs=rows.shape[1])
            t = ol.load(xin[0:M, 0:K])
            ol.store(y[0:M, 0:N], ol.dot(ol.quantize(t), w))

        from opentpu.runtime import compile_kernel, run
        comp, imgs = compile_kernel(lin, cfg, xin=Input(x), y=Output((3, 16)), M=3, N=16, K=512)
        for im in imgs:
            im[W_ROW:W_ROW + rows.size] = rows.reshape(-1)
            sw = CQ.scales(norms, s)
            im[S_ROW:S_ROW + sw.nbytes] = sw.view(np.uint8).reshape(-1)
        got = run(comp, imgs).outputs["y"]
        dev = x @ CQ.device_dense(codes, norms, cb).T
        assert np.linalg.norm(got - dev) / np.linalg.norm(dev) < .012, bits


def test_cq_pack_refuses_bad_codes():
    """Codes are checked before the byte cast (CQ3's 8, a 256 that would wrap to 0, a
    negative, a fraction)."""
    good = np.zeros((2, 128), np.int64)
    for bits, bad in ((3, 8), (4, 16), (2, 4), (1, 2), (4, 256), (4, -1)):
        c = good.copy()
        c[1, 5] = bad
        with pytest.raises(ValueError, match="codes"):
            CQ.pack(c, bits)
    with pytest.raises(ValueError, match="integers"):
        CQ.pack(good + 0.5, 4)
    assert CQ.pack(good.astype(np.float32) + 7, 3).shape == (2, 128)
