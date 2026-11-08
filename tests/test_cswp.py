"""VOP CSWP (Config.VCSWP, CAPS2 bit19): the bitonic compare-exchange stage and the sort on it
(kernels/sort.py)."""
import dataclasses

import numpy as np
import pytest

from opentpu import isa as I
from opentpu.isasim import Config, Machine, SimError
from opentpu.kernels import sort as S

A_T, B_T = 0, 0x4000


def cfg_(**kw):
    return Config(S=1, D=128, MCOLS=8, ACT_ROWS=16, ACT_BLOCKS=16, VCSWP=True, **kw)


def run(prog, x, cfg=None):
    cfg = cfg or cfg_()
    image = np.zeros(cfg.DRAM_BYTES, np.uint8)
    image[:x.nbytes] = x.view(np.uint8).reshape(-1)
    full = [I.ld(0, A_T, x.size)] + prog + [I.halt()]
    return cfg, image, full


def sample(rows, n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, n)).astype(np.float32)
    x[0, :3] = [0.0, -0.0, 0.0]                      # signed zeros
    x[-1, 1] = x[-1, 5]                              # a tie
    x[rows // 2, 2] = np.float32(np.inf)
    x[rows // 2, 4] = np.float32(-np.inf)
    return x


@pytest.mark.parametrize("rows,n,desc", [(3, 16, False), (2, 64, True), (5, 256, False),
                                         (1, 1024, True)])
def test_bitonic_sort_isa(rows, n, desc):
    x = sample(rows, n, rows + n)
    prog, out = S.bitonic(A_T, B_T, rows, n, n, descending=desc)
    cfg, image, full = run(prog, x)
    t = Machine(cfg, [full], [image]).run().slices[0].tmem
    got = t[out:out + rows * n].view(np.float32).reshape(rows, n)
    want = np.sort(x, axis=1)
    if desc:
        want = want[:, ::-1]
    np.testing.assert_array_equal(got, want)         # (+0 / -0 compare equal)


def test_cswp_one_stage_semantics():
    x = np.arange(16, dtype=np.float32)[::-1].copy().reshape(1, 16)
    prog = [I.cswp(B_T, A_T, 1, 16, 16, 16, 2, 8)]
    cfg, image, full = run(prog, x)
    got = Machine(cfg, [full], [image]).run().slices[0].tmem[B_T:B_T + 16].view(np.float32)
    c = np.arange(16)
    p = c ^ 2
    up = ((c & 2) != 0) != ((c & 8) != 0)
    want = np.where(up, np.maximum(x[0], x[0][p]), np.minimum(x[0], x[0][p]))
    np.testing.assert_array_equal(got, want)


def test_cswp_refusals():
    x = np.zeros((1, 16), np.float32)
    _, image, full = run([I.cswp(B_T, A_T, 1, 16, 16, 16, 4)], x)
    with pytest.raises(SimError, match="VCSWP"):
        Machine(dataclasses.replace(cfg_(), VCSWP=False), [full], [image]).run()
    with pytest.raises(ValueError):
        I.cswp(B_T, A_T, 1, 16, 16, 16, 3)
    with pytest.raises(ValueError):
        I.cswp(B_T, A_T, 1, 16, 16, 16, 4, 4)
    with pytest.raises(ValueError):
        I.cswp(B_T, A_T, 1, 12, 16, 16, 8)
    from opentpu.host.board import Board, ConfigMismatch
    from opentpu.host.fake import FakeTransport
    code = I.assemble(full)
    with pytest.raises(ConfigMismatch, match="VCSWP"):
        Board(FakeTransport(devname=None, regmap=5)).load_program(0x1000, code)
    Board(FakeTransport(devname=None, regmap=5, vcswp=True)).load_program(0x1000, code)


@pytest.mark.parametrize("rows,n,desc,waitc", [(3, 16, False, False), (2, 64, True, True),
                                               (2, 256, False, True)])
def test_bitonic_sort_rtl(have_verilator, rows, n, desc, waitc):
    from opentpu import rtlsim
    x = sample(rows, n, 7 + n)
    prog, out = S.bitonic(A_T, B_T, rows, n, n, descending=desc)
    cfg, image, full = run(prog, x, cfg_(SEQ_WC=waitc))
    ref = Machine(cfg, [full], [image.copy()]).run().slices[0]
    _, tmems, _ = rtlsim.run(cfg, [full], [image.copy()], axi=True, stall=20, seed=n)
    np.testing.assert_array_equal(tmems[0], ref.tmem)


@pytest.mark.parametrize("rows,n", [(2, 7), (3, 64), (1, 300)])
def test_cumsum_isa(rows, n):
    rng = np.random.default_rng(n)
    x = rng.uniform(0, 1, (rows, n)).astype(np.float32)
    prog, out = S.cumsum(A_T, B_T, rows, n, n)
    cfg, image, full = run(prog, x)
    t = Machine(cfg, [full], [image]).run().slices[0].tmem
    got = t[out:out + rows * n].view(np.float32).reshape(rows, n)
    np.testing.assert_array_equal(got, S.scan_ref(x))
    np.testing.assert_allclose(got, np.cumsum(x, 1), rtol=1e-5)
