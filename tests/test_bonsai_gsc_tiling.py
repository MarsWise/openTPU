"""Tiling is an experiment: it preserves numerics but does not cache weights."""
import dataclasses

import numpy as np
import pytest

from opentpu import isa as I
from opentpu.isasim import Machine
from opentpu.nextgen.gsc_tiling import tile_gsc
from test_tmm_gsc import case


@pytest.mark.parametrize('kb,m,acc', [(1, 1, False), (3, 6, True), (5, 1, True), (7, 6, False)])
@pytest.mark.parametrize('tile', [1, 3, 7, 10])
def test_tiles_preserve_gsc_output_bit_exact(kb, m, acc, tile):
    cfg, image, prog, *_ = case(kb, m, acc)
    command = prog[-2]
    command.flags &= ~I.F_RMAX
    tiled = tile_gsc(command, tile)
    orig = Machine(cfg, [prog], [image.copy()]).run().slices[0]
    exp = Machine(cfg, [prog[:-2] + tiled + [I.halt()]], [image.copy()]).run().slices[0]
    np.testing.assert_array_equal(exp.tmem, orig.tmem)
    np.testing.assert_array_equal(exp.dram, orig.dram)


def test_rmax_and_unsupported_inputs_are_refused():
    _, _, prog, *_ = case(kb=5, m=1)
    with pytest.raises(ValueError, match='RMAX'):
        tile_gsc(prog[-2], 3)
    with pytest.raises(ValueError, match='TMM GSC'):
        tile_gsc(I.halt(), 3)
    with pytest.raises(ValueError, match='positive'):
        tile_gsc(prog[-2], 0)


@pytest.mark.parametrize('waitc', [False, True])
def test_tiled_gsc_rtl_matches_original_isa(have_verilator, waitc):
    from opentpu import rtlsim
    cfg, image, prog, *_ = case(kb=5, m=1, acc=True)
    cfg = dataclasses.replace(cfg, SEQ_WC=waitc)
    prog[-2].flags &= ~I.F_RMAX
    ref = Machine(cfg, [prog], [image.copy()]).run().slices[0]
    tiled = prog[:-2] + tile_gsc(prog[-2], 3) + [I.halt()]
    drams, tmems, _ = rtlsim.run(cfg, [tiled], [image.copy()], axi=True, stall=45, seed=96)
    np.testing.assert_array_equal(tmems[0], ref.tmem)
    np.testing.assert_array_equal(drams[0], ref.dram)
