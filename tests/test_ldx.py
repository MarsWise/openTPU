"""LDX (Config.LDX, CAPS2 bit14): the DMA's indexed row gather from a DRAM table (docs/isa.md
"LDX"): deformable attention's samples, embedding rows, MoE expert rows. The ISA against numpy,
its refusals, the footprint, and RTL == ISA (rows of one / several / partial chunks, more rows
than the row queue, clamped and raw indices, DRAM stalls, a dependent VOP and a store before)."""
import dataclasses

import numpy as np
import pytest

from opentpu import isa as I
from opentpu import waits
from opentpu.isasim import Config, Machine, SimError

TAB, IDX, DST = 0x40000, 100, 1000


def cfg_d(D=128, **kw):
    base = dict(S=1, D=D, MCOLS=8 if D == 128 else 4, ACT_ROWS=16, ACT_BLOCKS=16,
                TMEM_WORDS=1 << 14, LDX=True)
    base.update(kw)
    return Config(**base)


def case(cfg, rows, cols, E, srs_chunks=None, drs=None, raw=False, seed=0, wild=False):
    """A table of E rows (srs bytes apart) and rows indices; the program LDXes them."""
    D = cfg.D
    rng = np.random.default_rng(seed + rows + 7 * cols)
    srs = (srs_chunks or -(-cols * 4 // D)) * D
    drs = drs if drs is not None else cols + 3
    tab = rng.normal(size=(E, srs // 4)).astype(np.float32)
    j = rng.integers(0, E, rows)
    if wild:                                           # out of range, both sides, and NaN
        j[::5] = E + 3
        j[1::7] = -2
    x = j.astype(np.int32).view(np.float32) if raw else j.astype(np.float32)
    x = x.copy()
    if wild and not raw:
        x[2] = np.nan
        x[3] = 0.7                                     # truncates to 0
    image = np.zeros(cfg.DRAM_BYTES, np.uint8)
    image[TAB:TAB + tab.nbytes] = tab.view(np.uint8).reshape(-1)
    image[0:x.nbytes] = x.view(np.uint8)
    prog = [I.ld(0, IDX, rows),
            I.ldx(TAB, DST, IDX, rows, cols, srs, drs, E, raw=raw, D=D),
            I.halt()]
    return image, prog, (tab, x, srs, drs)


def want(tab, x, rows, cols, E, raw):
    v = x.view(np.int32).astype(np.int64) if raw else \
        np.array([np.int32(np.uint32(I.f2i(int(b)))) for b in x.view(np.uint32)], np.int64)
    j = np.clip(v, 0, E - 1)
    return tab[j, :cols]


@pytest.mark.parametrize("rows,cols,E,raw,wild", [(9, 32, 40, False, False),
                                                  (20, 5, 7, True, False),
                                                  (33, 70, 50, False, True),
                                                  (3, 300, 4, True, True)])
def test_ldx_isa(rows, cols, E, raw, wild):
    cfg = cfg_d()
    image, prog, (tab, x, srs, drs) = case(cfg, rows, cols, E, raw=raw, wild=wild)
    s = Machine(cfg, [prog], [image]).run().slices[0]
    t = s.tmem.view(np.float32)
    got = np.stack([t[DST + i * drs:DST + i * drs + cols] for i in range(rows)])
    np.testing.assert_array_equal(got, want(tab, x, rows, cols, E, raw))


def test_ldx_refusals():
    cfg = cfg_d()
    image, prog, _ = case(cfg, 4, 8, 5)
    with pytest.raises(SimError, match="Config.LDX"):
        Machine(dataclasses.replace(cfg, LDX=False), [prog], [image]).run()
    with pytest.raises(ValueError, match="multiples of D"):
        I.ldx(TAB + 4, DST, IDX, 2, 8, 128, 8, 5)
    with pytest.raises(ValueError, match="multiples of D"):
        I.ldx(TAB, DST, IDX, 2, 8, 64, 8, 5)
    with pytest.raises(ValueError, match="cols"):
        I.ldx(TAB, DST, IDX, 2, 5000, 128 * 200, 8, 5)
    bad = [I.ld(0, IDX, 4), I.ldx(TAB, IDX + 2, IDX, 4, 8, 128, 8, 5), I.halt()]
    with pytest.raises(SimError, match="overlap"):
        Machine(cfg, [bad], [image]).run()


def test_ldx_footprint():
    """otpu_pkg's ranges: the table's E rows, the indices, the rows written."""
    ins = I.ldx(TAB, DST, IDX, 10, 40, 256, 50, 30)
    fp = waits.footprint(ins, [(0, 0)] * 16)
    assert (waits.SP_DRAM, TAB, TAB + 29 * 256 + 160) in fp.rd
    assert (waits.SP_TMEM, IDX, IDX + 10) in fp.rd
    assert (waits.SP_TMEM, DST, DST + 9 * 50 + 40) in fp.wr


def _prog_more(cfg, rows, cols, E, srs, drs):
    """A store into the table before the LDX (DRAM RAW) and a VOP on its rows after (TMEM RAW):
    the scoreboard orders them."""
    D = cfg.D
    return [I.ld(0, IDX, rows),
            I.ld(TAB, 4000, cols),                       # row 0 of the table, changed and stored
            I.vop(I.V_ADD, 4100, 4000, 4000, 1, cols, cols, cols, cols),
            I.st(TAB, 4100, cols),
            I.ldx(TAB, DST, IDX, rows, cols, srs, drs, E, D=D),
            I.vop(I.V_MUL, 6000, DST, DST, rows, cols, cols, drs, drs),
            I.halt()]


@pytest.mark.parametrize("D,rows,cols,E,raw,wild,stall", [
    (128, 9, 32, 40, False, False, 0),
    (128, 40, 5, 7, True, True, 35),
    (128, 21, 70, 50, False, True, 20),
    (32, 25, 13, 9, False, False, 30),
])
def test_ldx_rtl_matches_isa(have_verilator, D, rows, cols, E, raw, wild, stall):
    from opentpu import rtlsim
    cfg = cfg_d(D)
    image, prog, (tab, x, srs, drs) = case(cfg, rows, cols, E, raw=raw, wild=wild, seed=D)
    ref = Machine(cfg, [prog], [image.copy()]).run().slices[0]
    drams, tmems, _ = rtlsim.run(cfg, [prog], [image.copy()], axi=True, stall=stall, seed=rows)
    np.testing.assert_array_equal(drams[0], ref.dram)
    np.testing.assert_array_equal(tmems[0], ref.tmem)


def test_ldx_rtl_ordering(have_verilator):
    from opentpu import rtlsim
    cfg = cfg_d()
    rows, cols, E = 12, 24, 6
    image, _, (tab, x, srs, drs) = case(cfg, rows, cols, E)
    prog = _prog_more(cfg, rows, cols, E, srs, drs)
    ref = Machine(cfg, [prog], [image.copy()]).run().slices[0]
    t = ref.tmem.view(np.float32)
    j0 = np.flatnonzero(x == 0)
    if j0.size:                                          # row 0 doubled by the store
        np.testing.assert_array_equal(t[DST + j0[0] * drs:DST + j0[0] * drs + cols],
                                      2 * tab[0, :cols])
    drams, tmems, _ = rtlsim.run(cfg, [prog], [image.copy()], axi=True, stall=25, seed=3)
    np.testing.assert_array_equal(drams[0], ref.dram)
    np.testing.assert_array_equal(tmems[0], ref.tmem)
