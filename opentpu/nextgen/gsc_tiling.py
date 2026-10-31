"""Experimental output-row tiling for studying ternary GSC weight reuse.

Tiling alone does not cache weights or reduce their traffic. It is deliberately
not used by the production compiler. RMAX must be implemented across tiles
before this can cover that instruction; no PAIR/QUAD or binary mode is added.
"""
from dataclasses import replace

from .. import isa as I


def tile_gsc(ins: I.Instr, tile_rows: int) -> list[I.Instr]:
    """Split one GSC TMM by output columns, preserving each element's K order.

    The original runtime registers, output row stride, ACC and ACT ranges are
    preserved. Each instruction still performs the complete K loop for its
    output tile. Only valid for disjoint output columns and immutable weights
    while the original command would be in flight. There is no hardware cache.
    """
    if not isinstance(tile_rows, int) or tile_rows <= 0:
        raise ValueError('tile_rows must be positive')
    if ins.op != I.TMM or not ins.flags & I.F_GSC:
        raise ValueError('expected a TMM GSC instruction')
    if ins.flags & (I.F_RMAX | I.F_UNIT | I.F_RSC | I.F_PAIR | I.F_QUAD | I.F_ASCALE):
        raise ValueError('RMAX or incompatible GSC flags are unsupported')
    w = ins.w
    n, kb = w[3] & 0xffff, (w[3] >> 16) & 0xff
    ors = w[5] & 0xffff
    if not n or not kb or ors < n or w[3] >> 24:
        raise ValueError('invalid ternary GSC shape or reserved fields')
    out = []
    for row in range(0, n, tile_rows):
        count = min(tile_rows, n - row)
        part = list(w)
        part[0] = I.u32(w[0] + row * w[4])
        part[1] = I.u32(w[1] + row * w[6])
        part[2] = I.u32(w[2] + row)
        part[3] = (w[3] & 0xffff0000) | count
        out.append(replace(ins, w=part))
    return out
