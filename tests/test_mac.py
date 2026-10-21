"""VOP MAC (func 7): T[dst] = A * Cv(c) + B, Cv a column vector read first into the VPU's
column buffers, B in FULL / ROW / COL mode (in place into B too): the ISA simulator against
numpy, the RTL against the ISA bit for bit, and the kernels that use it (depthwise conv: one VOP
a tap; LayerNorm's affine step) on a target with the r5 vector extensions."""
import numpy as np
import pytest

import opentpu.language as ol
from opentpu import Config, fp32 as F, isa as I, rtlsim
from opentpu.isasim import Machine, SimError
from opentpu.kernels import conv as C
from opentpu.kernels.vit import layernorm_
from opentpu.runtime import Input, Output, compile_kernel, run

from conftest import assert_same_state


def _prog(cases, n):
    """cases: (bmode, rows, cols, inplace); A at 0, B at 4096, Cv at 8192, out at 12288 + ..."""
    prog = [I.ld(0, 0, n)]
    out = 12288
    for bm, rows, cols, inplace in cases:
        ars, brs = cols + 3, (cols + 5 if bm == I.B_FULL else 1 if bm == I.B_ROW else 0)
        if inplace:                            # B is the destination (an accumulator)
            prog.append(I.vop(I.V_COPY, out, 4096, 0, rows, cols, cols + 5, cols + 5, 0))
            prog.append(I.mac(out, 0, out, 8192, rows, cols, cols + 5, ars, cols + 5, I.B_FULL))
        else:
            prog.append(I.mac(out, 0, 4096, 8192, rows, cols, cols + 1, ars, brs, bm))
        prog.append(I.st(0x10000 + 4 * out, out, rows * (cols + 5)))
        out += rows * (cols + 5) + 7
    prog.append(I.halt())
    return prog


CASES = [(I.B_FULL, 3, 100, False), (I.B_ROW, 4, 37, False), (I.B_COL, 2, 256, False),
         (I.B_FULL, 5, 8, True), (I.B_COL, 1, 1, False), (I.B_FULL, 2, 200, True)]


def _image(rng, n):
    x = (rng.standard_normal(n) * 3).astype(np.float32)
    x[::17] = 0.0
    x[5] = -0.0
    dram = np.zeros(1 << 20, np.uint8)
    dram[:4 * n] = x.view(np.uint8)
    return dram


def test_mac_isa():
    rng = np.random.default_rng(0)
    n = 12288
    dram = _image(rng, n)
    x = dram[:4 * n].view(np.float32)
    rows, cols = 3, 100
    prog = [I.ld(0, 0, n), I.mac(20000, 0, 4096, 8192, rows, cols, cols, cols + 3, 9, I.B_COL),
            I.st(0x10000, 20000, rows * cols), I.halt()]
    m = Machine(Config(), [prog], [dram]).run()
    got = m.slices[0].dram[0x10000:0x10000 + 4 * rows * cols].view(np.float32).reshape(rows, cols)
    A = np.stack([x[r * (cols + 3):r * (cols + 3) + cols] for r in range(rows)])
    want = F.add(F.mul(A, x[8192:8192 + cols][None, :]), x[4096:4096 + cols][None, :])
    assert np.array_equal(F.bits(got), F.bits(want))
    with pytest.raises(SimError, match="MAC"):
        bad = [I.ld(0, 0, n), I.vop(I.V_MAC, 20000, 0, 0, 1, 8, 8, 8, 0, I.B_SCALAR, 0), I.halt()]
        Machine(Config(), [bad], [dram]).run()


@pytest.mark.parametrize("cfg", [Config(), Config(D=128, MCOLS=4, LANES=8)])
def test_mac_rtl(have_verilator, cfg):
    rng = np.random.default_rng(cfg.D)
    n = 12288
    dram = _image(rng, n)
    prog = _prog(CASES, n)
    m = Machine(cfg, [prog], [dram.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [dram.copy()])
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    assert np.array_equal(drams[0], m.slices[0].dram)


@ol.jit
def ln_k(x, w, b, y, M: int, N: int):
    t = ol.load(x)
    layernorm_(t, ol.load(w), ol.load(b), 1e-5)
    ol.store(y, t)


def test_mac_kernels_rtl(have_verilator):
    """Depthwise 3x3 (MAC a tap) and LayerNorm (MAC for the affine step) with the extensions:
    one VOP fewer each, the RTL equal to the ISA, the results against numpy."""
    rng = np.random.default_rng(4)
    cfg = Config(TFN=True)
    H, Cc, K = 6, 40, 3
    x = rng.standard_normal((H, H, Cc)).astype(np.float32)
    w = rng.standard_normal((Cc, 1, K, K)).astype(np.float32) / 3
    b = rng.standard_normal(Cc).astype(np.float32)
    args = dict(x=Input(C.pad_input_dw(x, 1)), w=Input(C.pack_dw_weight(w)), b=Input(b),
                y=Output((H * H, Cc)), H=H, W=H, C=Cc, K=K, stride=1, pad=1, act="none")
    n_vops = {}
    for c in (Config(), cfg):
        comp, _ = compile_kernel(C.depthwise2d, c, **args)
        n_vops[c.TFN] = sum(1 for p in comp.programs for i in p
                            if isinstance(i, I.Instr) and i.op == I.VOP)
    assert n_vops[True] < n_vops[False]
    comp, imgs = compile_kernel(C.depthwise2d, cfg, **args)
    ri = run(comp, [i.copy() for i in imgs], "isa")
    rr = run(comp, [i.copy() for i in imgs], "rtl")
    assert_same_state(ri, rr)
    xp = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    ref = np.zeros((H, H, Cc), np.float64) + b
    for kh in range(K):
        for kw in range(K):
            ref += xp[kh:kh + H, kw:kw + H] * w[:, 0, kh, kw]
    assert np.abs(rr.outputs["y"].reshape(H, H, Cc) - ref).max() < 1e-4

    M, N = 5, 300
    xx = rng.standard_normal((M, N)).astype(np.float32)
    ww, bb = rng.standard_normal(N).astype(np.float32), rng.standard_normal(N).astype(np.float32)
    largs = dict(x=Input(xx), w=Input(ww), b=Input(bb), y=Output((M, N)), M=M, N=N)
    comp, imgs = compile_kernel(ln_k, cfg, **largs)
    ri = run(comp, [i.copy() for i in imgs], "isa")
    rr = run(comp, [i.copy() for i in imgs], "rtl")
    assert_same_state(ri, rr)
    mu = xx.mean(1, keepdims=True)
    ref = (xx - mu) / np.sqrt(((xx - mu) ** 2).mean(1, keepdims=True) + 1e-5) * ww + bb
    assert np.abs(rr.outputs["y"] - ref).max() < 1e-4
