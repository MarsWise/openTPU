"""TMM T5 (Config.TD5, CAPS2 bit15): dense trits, five a byte (opentpu/trits.py: Bonsai 2's
PTQ1_0 / llama.cpp TQ1_0 byte coding), five D-element K blocks a D-byte chunk, with GSC's
group scales. The ISA simulator against the 2-bit TMM GSC of the same weights, the refusals, the
footprint (otpu_pkg's and opentpu.waits'), the compiler's dot."""
import dataclasses

import numpy as np
import pytest

from opentpu import isa as I
from opentpu import trits as T
from opentpu import waits
from opentpu.isasim import Config, Machine, SimError

D = 128
WA, SA, OUT = 0x10000, 0x8000, 4000


def _cfg(**kw):
    return Config(S=1, D=D, MCOLS=8, ACT_ROWS=16, ACT_BLOCKS=32, TMEM_WORDS=1 << 14,
                  TGSC=True, TD5=True, **kw)


def ternary2(w):
    """The 2-bit TMM rows (four blocks a chunk) of w [N, K]."""
    N, K = w.shape
    KB = K // D
    nc = -(-KB // 4)
    code = np.where(w == 1, 1, np.where(w == -1, 2, 0)).astype(np.uint8)
    sym = np.zeros((N, nc * 4 * D), np.uint8)
    sym[:, :K] = code
    sym = sym.reshape(N, nc * D, 4)
    return (sym[..., 0] | sym[..., 1] << 2 | sym[..., 2] << 4 | sym[..., 3] << 6).astype(np.uint8)


def run(N, KB, M, gsc, dense, seed=0):
    cfg = _cfg()
    rng = np.random.default_rng(seed + N + KB)
    K = KB * D
    w = rng.integers(-1, 2, (N, K))
    x = rng.normal(size=(M, K)).astype(np.float32)
    rows = T.pack(w, D) if dense else ternary2(w)
    rs = rows.shape[1]
    sw = rng.uniform(0.5, 2, (N, KB if gsc else 1)).astype(np.float32)
    srs = sw.shape[1] * 4
    img = np.zeros(cfg.DRAM_BYTES, np.uint8)
    img[WA:WA + rows.size] = rows.reshape(-1)
    img[SA:SA + sw.nbytes] = sw.view(np.uint8).reshape(-1)
    img[:x.nbytes] = x.view(np.uint8).reshape(-1)
    prog = [I.ld(0, 0, x.size), I.qact(0, M, 0, KB, K),
            I.tmm(WA, SA, OUT, N, KB, rs, N, M, 0, srs, rsc=not gsc, gsc=gsc, dense=dense),
            I.halt()]
    s = Machine(cfg, [prog], [img]).run().slices[0]
    return s.tmem[OUT:OUT + M * N].view(np.float32).reshape(M, N), (cfg, prog, img)


def test_the_byte_code_is_tq1_0s():
    t = np.array(np.meshgrid(*[range(3)] * 5, indexing="ij")).reshape(5, -1).T
    q = T.encode_bytes(t)
    assert len(set(q.tolist())) == 243 and np.array_equal(T.decode_bytes(q), t)
    assert T.encode_bytes([[0, 0, 0, 0, 0]])[0] == 0 and T.encode_bytes([[2] * 5])[0] == 255
    w = np.random.default_rng(1).integers(-1, 2, (3, 10 * D))
    assert np.array_equal(T.unpack(T.pack(w), 10 * D), w) and T.bits_a_weight(10 * D) == 1.6


@pytest.mark.parametrize("N,KB,M,gsc", [(5, 1, 1, True), (9, 7, 3, True), (16, 10, 8, True),
                                        (4, 12, 2, True)])
def test_t5_equals_the_2_bit_tmm(N, KB, M, gsc):
    a, _ = run(N, KB, M, gsc, True)
    b, _ = run(N, KB, M, gsc, False)
    np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))


def test_t5_refusals():
    _, (cfg, prog, img) = run(4, 3, 2, True, True)
    with pytest.raises(SimError, match="TD5"):
        Machine(dataclasses.replace(cfg, TD5=False), [prog], [img]).run()
    with pytest.raises(ValueError, match="T5"):
        I.tmm(WA, SA, OUT, 4, 2, D, 4, 2, 0, 4, rsc=True, dense=True)        # (GSC only)


def test_t5_footprint():
    ins = I.tmm(WA, SA, OUT, 6, 11, 3 * D, 6, 2, 0, 44, gsc=True, dense=True)
    fp = waits.footprint(ins, [(0, 0)] * 16)
    assert (waits.SP_DRAM, WA, WA + 5 * 3 * D + 3 * D) in fp.rd       # ceil(11 / 5) chunks


@pytest.mark.parametrize("gsc", [True])
def test_compiled_t5_linear(gsc):
    """ol.dot over QTensor.dense_weight: K split at five-block (chunk) boundaries when the
    operand's QACT ranges need it; equal to the same weights' 2-bit TMM, bit for bit."""
    from opentpu import language as ol
    from opentpu.compiler import QTensor
    from opentpu.runtime import Input, Output, compile_kernel, run as krun
    M, N, KB = 3, 16, 12
    K = KB * D
    rng = np.random.default_rng(3)
    w = rng.integers(-1, 2, (N, K))
    x = rng.normal(size=(M, K)).astype(np.float32)
    sw = rng.uniform(0.5, 2, (N, KB if gsc else 1)).astype(np.float32)
    outs = []
    for dense in (True, False):
        rows = T.pack(w, D) if dense else ternary2(w)

        @ol.jit
        def lin(xin, y, M: int, N: int, K: int):
            if dense:
                q = QTensor.dense_weight(WA, SA, N, K, D, rs=rows.shape[1], gsc=gsc)
            else:
                q = QTensor.ternary_weight(WA, SA, N, K, D, rs=rows.shape[1],
                                           srs=sw.shape[1] * 4, gsc=gsc)
            t = ol.load(xin[0:M, 0:K])
            ol.store(y[0:M, 0:N], ol.dot(ol.quantize(t), q))
        comp, imgs = compile_kernel(lin, _cfg(DRAM_BYTES=1 << 20), xin=Input(x),
                                    y=Output((M, N)), M=M, N=N, K=K)
        assert any("t5" in i.comment for p in comp.programs for i in p
                   if isinstance(i, I.Instr)) == dense
        for im in imgs:
            im[WA:WA + rows.size] = rows.reshape(-1)
            im[SA:SA + sw.nbytes] = sw.view(np.uint8).reshape(-1)
        outs.append(krun(comp, imgs).outputs["y"])
    np.testing.assert_array_equal(outs[0].view(np.uint32), outs[1].view(np.uint32))


@pytest.mark.parametrize("N,KB,M,stall", [(9, 7, 3, 0), (16, 11, 8, 30)])
def test_t5_rtl_matches_isa(have_verilator, N, KB, M, stall):
    """The GSC adapter's per-block subcommands (trit k mod 5 of chunk k / 5) and the MXU's T5
    decode on the RTL: DRAM and TMM equal the ISA simulator's."""
    from opentpu import rtlsim
    _, (cfg, prog, img) = run(N, KB, M, True, True, seed=7)
    ref = Machine(cfg, [prog], [img.copy()]).run().slices[0]
    drams, tmems, _ = rtlsim.run(cfg, [prog], [img.copy()], axi=True, stall=stall, seed=N)
    np.testing.assert_array_equal(tmems[0], ref.tmem)
    np.testing.assert_array_equal(drams[0], ref.dram)
