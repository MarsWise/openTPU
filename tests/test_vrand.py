"""VOP RAND (opentpu/rand.py, rtl/vpu/otpu_rand.sv; Config.VRAND, CAPS2 bit10): the
counter-based uniforms the device draws itself (no host random numbers: docs/nextgen r8)."""
import dataclasses

import numpy as np
import pytest

from opentpu import Config, isa as I, rtlsim
from opentpu import rand as RN
from opentpu.isasim import Machine, SimError
from opentpu.target import arch_hash

# Random123's known-answer vectors (kat_vectors: threefry2x32 13 and 20 rounds)
KAT = [((0, 0), (0, 0), 13, (0x9d1c5ec6, 0x8bd50731)),
       ((0, 0), (0, 0), 20, (0x6b200159, 0x99ba4efe)),
       ((0xffffffff, 0xffffffff), (0xffffffff, 0xffffffff), 13, (0xfd36d048, 0x2d17272c)),
       ((0xffffffff, 0xffffffff), (0xffffffff, 0xffffffff), 20, (0x1cb996fc, 0xbb002be7)),
       ((0x243f6a88, 0x85a308d3), (0x13198a2e, 0x03707344), 13, (0xba3e4725, 0xf27d669e)),
       ((0x243f6a88, 0x85a308d3), (0x13198a2e, 0x03707344), 20, (0xc4923a9c, 0x483df7a0))]


@pytest.mark.parametrize("ctr, key, rounds, want", KAT)
def test_threefry_known_answers(ctr, key, rounds, want):
    got = RN.threefry2x32(np.uint32(ctr[0]), np.uint32(ctr[1]), np.uint32(key[0]),
                          np.uint32(key[1]), rounds)
    assert (int(got[0]), int(got[1])) == want


def test_uniforms_are_exact_and_uniform():
    u = RN.uniform(RN.iota(1 << 16), 12345)
    assert u.dtype == np.float32 and u.min() >= 0 and u.max() <= 1 - 2.0 ** -24
    assert np.all(u * np.float32(2 ** 24) == np.floor(u * np.float32(2 ** 24)))
    # moments and a chi-square over 64 bins (2^16 draws): loose, deterministic
    assert abs(u.mean() - 0.5) < 0.005 and abs(u.var() - 1 / 12) < 0.002
    h = np.bincount((u * 64).astype(int), minlength=64)
    chi2 = ((h - len(u) / 64) ** 2 / (len(u) / 64)).sum()
    assert chi2 < 120                                    # 63 dof: p ~ 1e-5 above
    # keys and counters decorrelate: neighbouring keys differ everywhere
    v = RN.uniform(RN.iota(1 << 16), 12346)
    assert abs(np.corrcoef(u, v)[0, 1]) < 0.02 and np.mean(u == v) < 1e-3


def cfg(waitc=False):
    return Config(D=128, MCOLS=8, ACT_BLOCKS=16, TMEM_WORDS=4096, DRAM_BYTES=1 << 20,
                  VRAND=True, SEQ_WC=waitc)


def program(c, n, rows, cols, drs, ars, key, rd_key=None, mix=False):
    """LD n counter words to T[0..n), RAND rows x cols from T[0] (ars) to T[2048] (drs), ST.
    rd_key: the key's register part (R5 = rd_key, w7 = key). mix: an ADD and an EXP2 in flight
    around RAND (the pipeline's other latencies)."""
    out = [I.ld(0, 0, n)]
    if rd_key is not None:
        out.append(I.li(5, rd_key))
    if mix:
        out.append(I.vop(I.V_EXP2, 3000, 0, 0, 1, 40, 0, 0, 0))
        out.append(I.vop(I.V_ADD, 3100, 0, 0, 1, 40, 0, 0, 0, I.B_SCALAR, 1.0))
    out.append(I.rand(2048, 0, rows, cols, drs, ars, key, rd=5 if rd_key is not None else 0))
    if mix:
        out.append(I.vop(I.V_MUL, 3200, 2048, 0, 1, 33, 0, 0, 0, I.B_SCALAR, 3.0))
    out.append(I.st(0x40000, 2048, rows * max(drs, cols)))
    out.append(I.st(0x50000, 3000, 300))
    out.append(I.halt())
    return out


def image(c, n, base=0):
    img = np.zeros(c.DRAM_BYTES, np.uint8)
    w = RN.iota(n, base)
    sp = np.array([0, 0xffffffff, 0x7fc00000, 0x80000000], np.uint32)  # NaN / -0 patterns
    w[:min(4, n)] = sp[:min(4, n)]                      # (raw counters)
    img[:4 * n] = w.view(np.uint8)
    return img, w


@pytest.mark.parametrize("rows, cols, drs, ars", [(1, 1, 1, 1), (1, 37, 37, 37),
                                                  (3, 19, 24, 21), (4, 64, 64, 64)])
def test_rand_isa_matches_the_reference(rows, cols, drs, ars):
    c = cfg()
    n = (rows - 1) * ars + cols
    img, w = image(c, n, 1000)
    s = Machine(c, [program(c, n, rows, cols, drs, ars, 0xDEADBEEF)], [img]).run().slices[0]
    got = s.tmem[2048:2048 + rows * drs].view(np.float32).reshape(rows, drs)[:, :cols]
    ctr = np.stack([w[r * ars:r * ars + cols] for r in range(rows)])
    np.testing.assert_array_equal(got, RN.uniform(ctr, 0xDEADBEEF))


def test_rand_key_is_the_register_plus_the_immediate():
    c = cfg()
    img, w = image(c, 50)
    s = Machine(c, [program(c, 50, 1, 50, 50, 50, 7, rd_key=0xFFFFFFFE)], [img]).run().slices[0]
    got = s.tmem[2048:2098].view(np.float32)
    np.testing.assert_array_equal(got, RN.uniform(w, 5))      # (2^32 - 2 + 7) mod 2^32


@pytest.mark.parametrize("rows, cols, drs, ars, mix, waitc", [
    (1, 1, 1, 1, False, False), (1, 37, 37, 37, True, False), (3, 19, 24, 21, True, True),
    (4, 64, 64, 64, False, True)])
def test_rand_rtl_matches_isa_with_axi_stalls(have_verilator, rows, cols, drs, ars, mix, waitc):
    c = cfg(waitc)
    n = (rows - 1) * ars + cols
    img, _ = image(c, max(n, 40), 77)
    prog = program(c, max(n, 40), rows, cols, drs, ars, 0x12345678, rd_key=0x0F0F0F0F, mix=mix)
    ref = Machine(c, [prog], [img.copy()]).run().slices[0]
    drams, tmems, _ = rtlsim.run(c, [prog], [img.copy()], axi=True, stall=35, seed=rows + cols)
    np.testing.assert_array_equal(tmems[0], ref.tmem)
    np.testing.assert_array_equal(drams[0], ref.dram)


def test_rand_capability():
    from opentpu.host.board import Board, ConfigMismatch, device_config
    from opentpu.host.fake import FakeTransport
    c = cfg()
    img, _ = image(c, 8)
    prog = program(c, 8, 1, 8, 8, 8, 1)
    old = dataclasses.replace(c, VRAND=False)
    assert arch_hash(old) != arch_hash(c)
    with pytest.raises(SimError, match="Config.VRAND"):
        Machine(old, [prog], [img]).run()
    code = I.assemble(prog)
    with pytest.raises(ConfigMismatch, match="VRAND"):
        Board(FakeTransport(devname=None, regmap=5)).load_program(0x1000, code)
    board = Board(FakeTransport(devname=None, regmap=5, vrand=True))
    assert device_config(board.info()).VRAND
    board.load_program(0x1000, code)
