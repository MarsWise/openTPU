"""QST Q6 (Config.QST6, CAPS2 bit18): the 6-bit K cache as three 2-bit planes in TMM CQ2's GRQ
layout (kernels/kv6.py), the scores as three TMM CQ2 GRQs a block."""
import dataclasses

import numpy as np
import pytest

from opentpu import fp32 as F
from opentpu import isa as I
from opentpu.isasim import Config, Machine, SimError
from opentpu.kernels import kv6

D = 128
Q_T, K_T, OUT = 0, 0x2000, 0x8000             # TMEM words
PL, SC = 0x40000, 0x7F000                     # DRAM bytes


def cfg_(**kw):
    return Config(S=1, D=D, MCOLS=8, ACT_ROWS=16, ACT_BLOCKS=16, TGSC=True, TBIN=True,
                  TBGSC=True, TCQ=True, TGRQ=True, QST6=True, **kw)


def case(n=12, kb=2, m=3, N=16, seed=0, row0=0, waitc=False):
    cfg = cfg_(SEQ_WC=waitc)
    rng = np.random.default_rng(60 + 7 * n + kb + m + seed)
    K = rng.normal(size=(n, kb * D)).astype(np.float32)
    K[0, :5] = 0.0
    if n > 2:
        K[2] *= 40.0                                 # a large row
        K[1, kb * D - 3] = 0.0
    q = rng.normal(size=(m, kb * D)).astype(np.float32)
    image = np.zeros(cfg.DRAM_BYTES, np.uint8)
    image[:q.nbytes] = q.view(np.uint8).reshape(-1)
    image[0x8000:0x8000 + K.nbytes] = K.view(np.uint8).reshape(-1)
    prog = [I.ld(0, Q_T, q.size), I.ld(0x8000, K_T, K.size), I.qact(Q_T, m, 0, kb, kb * D)]
    prog += kv6.store(K_T, kb * D, row0, n, kb, N, PL, SC)
    prog += kv6.scores(row0 + n, kb, N, PL, SC, OUT, N + 1, m)
    prog.append(I.halt())
    return cfg, image, prog, (K, q, N)


def quant_ref(K):
    """The 6-bit codes and S of each (row, block) as kv6 stores them."""
    n, kbD = K.shape
    blocks = K.reshape(n, kbD // D, D).transpose(1, 0, 2)
    return [F.quantize_q4(b, 31) for b in blocks]


@pytest.mark.parametrize("n,kb,m", [(12, 2, 3), (5, 1, 8), (9, 3, 1)])
def test_qst6_planes_and_scores_isa(n, kb, m):
    cfg, image, prog, (K, q, N) = case(n, kb, m)
    s = Machine(cfg, [prog], [image]).run().slices[0]
    Kd = kv6.dequant(s.dram, n, kb, N, PL, SC)
    for k, (qq, S) in enumerate(quant_ref(K)):
        np.testing.assert_array_equal(Kd[:, k * D:(k + 1) * D], qq.astype(np.float32) * S[:, None])
        assert np.abs(qq).max() <= 31
    # 6 bits: within S/2 of K
    S_all = np.stack([S for _, S in quant_ref(K)], 1)
    assert np.all(np.abs(Kd - K) <= np.repeat(S_all, D, 1) * .5001 + 1e-30)
    got = s.tmem[OUT:OUT + m * (N + 1)].view(np.float32).reshape(m, N + 1)[:, :n]
    want = q @ Kd.T
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < .012
    want_f = q @ K.T
    # (the 6-bit K's own error, |K - Kd| <= S/2 above, is a few % of one dot)
    assert np.linalg.norm(got - want_f) / np.linalg.norm(want_f) < .1


def test_qst6_bytes_layout():
    """Plane j: four codes ((q + 32) >> 2j) & 3 a byte, element c of row r at r*D/4 + c/4."""
    cfg, image, prog, (K, q, N) = case(3, 1, 1)
    s = Machine(cfg, [prog], [image]).run().slices[0]
    (qq, S), = quant_ref(K)
    for j in range(3):
        base = kv6.plane(PL, N, 0, j)
        for r in range(3):
            b = s.dram[base + r * D // 4:base + (r + 1) * D // 4]
            for c in (0, 1, 2, 3, 5, 77, 127):
                assert (b[c // 4] >> (2 * (c % 4))) & 3 == ((int(qq[r, c]) + 32) >> (2 * j)) & 3
        assert not s.dram[base + 3 * D // 4:base + N * D // 4].any()   # rows 3.. untouched
    for r in range(3):
        assert s.dram[SC + 4 * r:SC + 4 * r + 4].view(np.float32)[0] == S[r]


def test_qst6_append_rows():
    """Decode: one row appended at row0 = 7 (the earlier rows' planes untouched)."""
    cfg, image, prog, (K, q, N) = case(1, 2, 2, row0=7)
    s = Machine(cfg, [prog], [image]).run().slices[0]
    for k in range(2):
        for j in range(3):
            assert not s.dram[kv6.plane(PL, N, k, j):kv6.plane(PL, N, k, j) + 7 * D // 4].any()
    Kd = kv6.dequant(s.dram, 8, 2, N, PL, SC)
    assert not Kd[:7].any()
    np.testing.assert_allclose(Kd[7], K[0], atol=float(np.abs(K[0]).max() / 31 * .51))


def test_qst6_refusals():
    cfg, image, prog, _ = case(4, 1, 1)
    with pytest.raises(SimError, match="QST6"):
        Machine(dataclasses.replace(cfg, QST6=False), [prog], [image]).run()
    with pytest.raises(ValueError):
        I.qst(0, 0, 0, 1, 1, 128, 64, 1, row=True, q6=3)
    with pytest.raises(ValueError):
        I.qst(0, 0, 0, 1, 1, 128, 64, 1, row=False, q6=0)
    with pytest.raises(ValueError):
        I.qst(0, 0, 0, 1, 1, 128, 64, 1, row=True, q4=True, q6=1)
    with pytest.raises(ValueError):
        I.tmm_cq(PL, SC, 0, 4, 1, 128, 4, 1, 0, 16, 4, grq=True)
    bad = [I.qst(K_T, PL + 2, SC, 1, 1, D, 32, 1, row=True, q6=2), I.halt()]
    with pytest.raises(SimError, match="word-aligned"):
        Machine(cfg, [prog[:2] + bad], [image]).run()
    from opentpu.host.board import Board, ConfigMismatch
    from opentpu.host.fake import FakeTransport
    code = I.assemble(prog)
    kw = dict(devname=None, regmap=5, tgsc=True, tbin=True, tbgsc=True, tcq=True, tgrq=True)
    with pytest.raises(ConfigMismatch, match="QST6"):
        Board(FakeTransport(**kw)).load_program(0x1000, code)
    Board(FakeTransport(**kw, qst6=True)).load_program(0x1000, code)


@pytest.mark.parametrize("n,kb,m,waitc", [(12, 2, 3, False), (5, 1, 8, True), (9, 3, 1, True)])
def test_qst6_rtl_matches_isa(have_verilator, n, kb, m, waitc):
    from opentpu import rtlsim
    cfg, image, prog, _ = case(n, kb, m, waitc=waitc)
    ref = Machine(cfg, [prog], [image.copy()]).run().slices[0]
    drams, tmems, _ = rtlsim.run(cfg, [prog], [image.copy()], axi=True, stall=35, seed=n + kb)
    np.testing.assert_array_equal(drams[0], ref.dram)
    np.testing.assert_array_equal(tmems[0], ref.tmem)


def test_qst6_append_rtl(have_verilator):
    from opentpu import rtlsim
    cfg, image, prog, _ = case(1, 2, 2, row0=7)
    ref = Machine(cfg, [prog], [image.copy()]).run().slices[0]
    drams, tmems, _ = rtlsim.run(cfg, [prog], [image.copy()], axi=True, stall=20, seed=3)
    np.testing.assert_array_equal(drams[0], ref.dram)
    np.testing.assert_array_equal(tmems[0], ref.tmem)


# ---- the 6-bit V^T cache (QST Q6T): groups of four tokens a byte, S per token folded into P
V_T, P_T, S_T, PO = 0, 0x4400, 0x5000, 0x5200          # TMEM words
VPL, VSC, ONES = 0x50000, 0x7E000, 0x7D000             # DRAM bytes


def vcase(n=6, H=128, m=2, appends=(), seed=0, waitc=False):
    """Tokens 0..n-1 by one QST batch, then each `appends` token t rewriting its group."""
    cfg = cfg_(SEQ_WC=waitc)
    rng = np.random.default_rng(300 + n + 7 * m + seed)
    nt = n + len(appends)
    V = rng.normal(size=(nt, H)).astype(np.float32)
    V[1] *= 20.0
    nb = -(-nt // D)
    P = np.zeros((m, nb * D), np.float32)
    P[:, :nt] = rng.uniform(0, 1, (m, nt)).astype(np.float32)
    image = np.zeros(cfg.DRAM_BYTES, np.uint8)
    image[0x8000:0x8000 + V.nbytes] = V.view(np.uint8).reshape(-1)
    image[0x20000:0x20000 + P.nbytes] = P.view(np.uint8).reshape(-1)
    image[ONES:ONES + 4 * H] = np.ones(H, np.float32).view(np.uint8)
    prog = [I.ld(0x8000, V_T, V.size), I.ld(0x20000, P_T, P.size)]
    prog += kv6.store_v(V_T, H, 0, n, H, VPL, VSC)
    for t in appends:
        g = t - t % 4
        prog += kv6.store_v(V_T + g * H, H, g, t - g + 1, H, VPL, VSC)
    prog += [I.ld(VSC, S_T, nt),
             I.vop(I.V_MUL, P_T, P_T, S_T, m, nt, nb * D, nb * D, 0, I.B_COL),
             I.qact(P_T, m, 0, nb, nb * D)]
    prog += kv6.pv(nt, H, VPL, ONES, PO, H, m)
    prog.append(I.halt())
    return cfg, image, prog, (V, P, nt)


@pytest.mark.parametrize("n,m,appends", [(6, 2, ()), (5, 1, (5, 6, 7, 8)), (130, 3, (130,))])
def test_qst6_v_cache_isa(n, m, appends):
    cfg, image, prog, (V, P, nt) = vcase(n, m=m, appends=appends)
    s = Machine(cfg, [prog], [image]).run().slices[0]
    Vd = kv6.dequant_v(s.dram, nt, 128, VPL, VSC)
    qq, S = F.quantize_q4(V, 31)
    np.testing.assert_array_equal(Vd, qq.astype(np.float32) * S[:, None])
    got = s.tmem[PO:PO + m * 128].view(np.float32).reshape(m, 128)
    want = P[:, :nt] @ Vd
    # (P' = P * S takes token 1's 20x S: its int8 QACT, not the planes, sets the error)
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < .02
    want_f = P[:, :nt] @ V
    assert np.linalg.norm(got - want_f) / np.linalg.norm(want_f) < .05


def test_qst6_v_partial_group_bytes():
    """A group of two tokens: the missing tokens' bits are q = 0's code (u 32: plane 2 code 2)."""
    cfg, image, prog, (V, P, nt) = vcase(2, m=1)
    s = Machine(cfg, [prog], [image]).run().slices[0]
    qq, _ = F.quantize_q4(V, 31)
    for j in range(3):
        b = s.dram[kv6.vplane(VPL, 128, 0, j) + np.arange(128) * 32]
        for r in range(4):
            want = ((qq[r].astype(int) + 32) >> (2 * j)) & 3 if r < 2 else (2 if j == 2 else 0)
            np.testing.assert_array_equal((b >> (2 * r)) & 3, want)
    with pytest.raises(ValueError):
        I.qst(0, 0, 0, 2, 1, 128, 2, 32, row=True, q6=0, q6t=True)      # drs 1 only


@pytest.mark.parametrize("n,m,appends,waitc", [(6, 2, (), False), (5, 1, (5, 6, 7, 8), True),
                                               (130, 3, (130,), True)])
def test_qst6_v_cache_rtl(have_verilator, n, m, appends, waitc):
    from opentpu import rtlsim
    cfg, image, prog, _ = vcase(n, m=m, appends=appends, waitc=waitc)
    ref = Machine(cfg, [prog], [image.copy()]).run().slices[0]
    drams, tmems, _ = rtlsim.run(cfg, [prog], [image.copy()], axi=True, stall=30, seed=n + m)
    np.testing.assert_array_equal(drams[0], ref.dram)
    np.testing.assert_array_equal(tmems[0], ref.tmem)
