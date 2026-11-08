"""Bitonic sort of TMEM rows on the VPU (VOP CSWP, Config.VCSWP): log2(n) (log2(n) + 1) / 2
compare-exchange stages a row block, each one pass over the rows (n a power of two; pad with
-inf / +inf). Used for the top-k / top-p thresholds of sampling on the card: the sorted row's
k-th value (or the cutoff where the sorted cumulative mass passes p) masks the unsorted row with
CMP / SEL, so no indices travel with the values."""
from __future__ import annotations

from opentpu import isa as I


def stages(n: int):
    """The (s, k) pairs of a bitonic sort of n (a power of two): k = 0 on the last merge."""
    assert n >= 2 and not n & (n - 1)
    out = []
    k = 2
    while k <= n:
        s = k // 2
        while s >= 1:
            out.append((s, 0 if k == n else k))
            s //= 2
        k *= 2
    return out


def bitonic(src, tmp, rows, n, rs, descending=False):
    """Sort each of `rows` rows of n words at src (row stride rs) ascending (descending: of
    -x, negated back). The result is at the returned address (src or tmp, the same layout)."""
    prog = []
    a, b = src, tmp
    if descending:
        prog.append(I.vop(I.V_MUL, b, a, 0, rows, n, rs, rs, 0, I.B_SCALAR, -1.0,
                          comment="sort: -x"))
        a, b = b, a
    for s, k in stages(n):
        prog.append(I.cswp(b, a, rows, n, rs, rs, s, k, comment=f"bitonic s{s} k{k}"))
        a, b = b, a
    if descending:
        prog.append(I.vop(I.V_MUL, b, a, 0, rows, n, rs, rs, 0, I.B_SCALAR, -1.0,
                          comment="sort: -x back"))
        a, b = b, a
    return prog, a


def cumsum(src, tmp, rows, n, rs):
    """Inclusive prefix sums of each row (Hillis-Steele, log2(n) stages of two VOPs: COPY of
    the first s columns, ADD of column c - s to c >= s; existing VOPs, no new unit). Returns
    (prog, the address of the result). The fp32 additions run in the stages' order."""
    prog, a, b, s = [], src, tmp, 1
    while s < n:
        prog.append(I.vop(I.V_COPY, b, a, 0, rows, s, rs, rs, 0, comment=f"scan s{s} copy"))
        prog.append(I.vop(I.V_ADD, b + s, a + s, a, rows, n - s, rs, rs, rs,
                          comment=f"scan s{s} add"))
        a, b, s = b, a, 2 * s
    return prog, a


def scan_ref(x):
    """The fp32 result cumsum computes (the same stage order)."""
    import numpy as np
    from opentpu import fp32 as F
    y = x.astype(np.float32).copy()
    s = 1
    while s < y.shape[1]:
        z = y.copy()
        z[:, s:] = F.add(y[:, s:], y[:, :-s])
        y, s = z, 2 * s
    return y
