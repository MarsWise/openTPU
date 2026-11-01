"""QST Q4 (Config.QST4, CAPS2 bit13): 4-bit ROW rows, two elements a byte with an RSC scale
word, the operand of an int4 MM with RSC: a 4-bit K cache (Hadamard-rotated Q and K keep its
error low: kernels/hadamard.py)."""
import dataclasses

import numpy as np
import pytest

from opentpu import fp32 as F
from opentpu import isa as I
from opentpu.isasim import Config, Machine, SimError

KD, KS, OUT = 0x20000, 0x30000, 6000


def case(rows=9, kb=1, m=2, waitc=False, seed=0):
    cfg = Config(S=1, D=128, MCOLS=8, ACT_ROWS=16, ACT_BLOCKS=16, TMEM_WORDS=1 << 14,
                 QST4=True, SEQ_WC=waitc)
    D = cfg.D
    rng = np.random.default_rng(40 + rows + kb + seed)
    k = rng.normal(size=(rows, kb * D)).astype(np.float32)
    k[0, :] = 0                                     # a zero row: S 0, codes 0
    q = rng.normal(size=(m, kb * D)).astype(np.float32)
    image = np.zeros(cfg.DRAM_BYTES, np.uint8)
    image[:k.nbytes] = k.view(np.uint8).reshape(-1)
    image[0x10000:0x10000 + q.nbytes] = q.view(np.uint8).reshape(-1)
    rs = kb * D // 2 + 64                           # rows apart by more than their bytes
    rs = -(-rs // D) * D
    prog = [I.ld(0, 0, k.size), I.ld(0x10000, 4096, q.size),
            I.qst(0, KD, KS, rows, kb, kb * D, rs, 1, row=True, q4=True),
            I.qact(4096, m, 0, kb, kb * D),
            I.mm(KD, KS, OUT, rows, kb, rs, rows + 1, m, 0, 4, wf=I.W4I, rsc=True),
            I.halt()]
    return cfg, image, prog, (k, q, rs)


@pytest.mark.parametrize("rows,kb,m", [(9, 1, 2), (4, 2, 1), (16, 1, 8)])
def test_q4_rows_and_the_int4_mm(rows, kb, m):
    cfg, image, prog, (k, q, rs) = case(rows, kb, m)
    s = Machine(cfg, [prog], [image]).run().slices[0]
    qk, sk = F.quantize_q4(k)
    raw = s.dram[KD:KD + rows * rs].reshape(rows, rs)[:, :kb * 64]
    lo = (raw & 15).astype(np.int8)
    hi = (raw >> 4).astype(np.int8)
    got = np.empty_like(qk)
    got[:, 0::2] = np.where(lo > 7, lo - 16, lo)
    got[:, 1::2] = np.where(hi > 7, hi - 16, hi)
    np.testing.assert_array_equal(got, qk)
    np.testing.assert_array_equal(s.dram[KS:KS + 4 * rows].view(np.float32), sk)
    assert sk[0] == 0 and np.all(qk[0] == 0)
    y = s.tmem[OUT:OUT + m * (rows + 1)].view(np.float32).reshape(m, rows + 1)[:, :rows]
    want = q @ (qk.astype(np.float32) * sk[:, None]).T
    assert np.linalg.norm(y - want) / np.linalg.norm(want) < .02     # the activations' int8


def test_q4_refusals():
    cfg, image, prog, _ = case()
    with pytest.raises(SimError, match="QST4"):
        Machine(dataclasses.replace(cfg, QST4=False), [prog], [image]).run()
    with pytest.raises(ValueError, match="Q4"):
        I.qst(0, KD, KS, 2, 1, 128, 64, 2, row=True, q4=True)
    with pytest.raises(ValueError, match="Q4"):
        I.qst(0, KD, KS, 2, 1, 128, 64, 1, row=False, q4=True)


@pytest.mark.parametrize("rows,kb,m,waitc", [(9, 1, 2, False), (16, 2, 8, True)])
def test_q4_rtl_matches_isa(have_verilator, rows, kb, m, waitc):
    from opentpu import rtlsim
    cfg, image, prog, _ = case(rows, kb, m, waitc)
    ref = Machine(cfg, [prog], [image.copy()]).run().slices[0]
    drams, tmems, _ = rtlsim.run(cfg, [prog], [image.copy()], axi=True, stall=35, seed=rows)
    np.testing.assert_array_equal(drams[0], ref.dram)
    np.testing.assert_array_equal(tmems[0], ref.tmem)
