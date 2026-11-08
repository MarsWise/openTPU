"""6-bit K cache (Config.QST6 + TCQ + TGRQ): QST Q6 writes each D-element block of a K row as
three 2-bit planes, the codes ((q + 32) >> 2j) & 3 of q = clamp(rne(x * 31/amax), -31, 31),
with one fp32 S = amax/31 a (row, block). Each plane is TMM CQ2's GRQ layout (four rows a D-byte
chunk: row n's D/4 bytes at n*D/4, every chunk read once), so the scores Q.K^T are three TMM
CQ2 GRQs a block, plane j with the LUT 4^j c (j < 2) or 16 c - 32 (j = 2), accumulating:
    q = c0 + 4 c1 + 16 c2 - 32,   K[n, k*D + i] ~ q * S[k, n].
6 bits of DRAM a value (int8: 8) and three chunk reads per four rows (MM W8: four).

Layout (block k, plane j, of a cache of N rows, N % 4 == 0):
    pl + (3k + j) * N*D/4 + n*D/4   row n's codes (D/4 bytes)
    sc + k*N*4 + 4n                 S (16-byte aligned planes of N words)
"""
from __future__ import annotations

import numpy as np

from opentpu import isa as I

LUTS = [[c for c in range(4)], [4 * c for c in range(4)], [16 * c - 32 for c in range(4)]]


def plane(pl, N, k, j, D=128):
    return pl + (3 * k + j) * N * D // 4


def store(src, srs, row0, rows, kb, N, pl, sc, D=128, ra=0):
    """QSTs of rows row0..row0+rows-1 (TMEM src + r*srs, kb*D fp32 each) into the cache."""
    assert N % 4 == 0 and sc % 16 == 0 and pl % D == 0
    out = []
    for k in range(kb):
        for j in range(3):
            out.append(I.qst(src + k * D, plane(pl, N, k, j, D) + row0 * D // 4,
                             sc + k * N * 4 + 4 * row0, rows, 1, srs, D // 4, 1, row=True, q6=j,
                             ra=ra, comment=f"kv6 k{k} plane {j}"))
    return out


def scores(n, kb, N, pl, sc, out, ors, m, ab=0, acc=False, D=128):
    """The [m, n4] scores (n4 = n rounded up to 4; the rows past n are the cache's next rows)
    of the QACT'ed queries (ACT rows 0..m-1, blocks ab..ab+kb-1) against the cache's rows into
    TMEM out (row stride ors >= n4); acc: added to what out holds."""
    n4 = -(-n // 4) * 4
    assert n4 <= N and ors >= n4
    prog, first = [], not acc
    for j in range(3):
        prog += I.tmm_lut(LUTS[j], f"lutset kv6 plane {j}")
        for k in range(kb):
            # GRQ: the TMM's plane stride rs is unused with KB 1; the scale plane at sc + k*N*4
            prog.append(I.tmm_cq(plane(pl, N, k, j, D), sc + k * N * 4, out, n4, 1, N * D // 4,
                                 ors, m, ab + k, 4 * N, 2, acc=not first, grq=True,
                                 comment=f"kv6 k{k} plane {j}"))
            first = False
    return prog


def dequant(dram, n, kb, N, pl, sc, D=128):
    """The fp32 [n, kb*D] K the planes stand for (q * S), from a DRAM image."""
    K = np.zeros((n, kb * D), np.float32)
    for k in range(kb):
        s = dram[sc + k * N * 4:sc + k * N * 4 + 4 * n].copy().view(np.float32)
        q = np.full((n, D), -32, np.int32)
        for j in range(3):
            b = dram[plane(pl, N, k, j, D):plane(pl, N, k, j, D) + n * D // 4].reshape(n, D // 4)
            c = np.stack([(b >> (2 * t)) & 3 for t in range(4)], 2).reshape(n, D).astype(np.int32)
            q += c << (2 * j)
        K[:, k * D:(k + 1) * D] = q.astype(np.float32) * s[:, None]
    return K


# ---- the 6-bit V cache: V^T (head dim rows, token columns) as the same three planes, written a
# group of four tokens a byte (QST Q6T) with S per token, which P absorbs (P'[t] = P[t] * S[t])
#   vpl + (3b + j) * H*D/4 + c*D/4 + (t % D)/4   token t's codes of dim c, bits 2*(t % 4)
#   vsc + 4t                                     S of token t
# (H the head dim rows, H % 4 == 0; b = t // D the token block)

def vplane(vpl, H, b, j, D=128):
    return vpl + (3 * b + j) * H * D // 4


def store_v(src, srs, t0, rows, H, vpl, vsc, D=128, ra=0):
    """QST Q6Ts of tokens t0..t0+rows-1 (TMEM src + r*srs, H fp32 each; t0 % 4 == 0): every
    group's byte is written whole (a group's missing tokens as q 0), so a decode append
    rewrites its group from the group's first token."""
    assert t0 % 4 == 0 and H % D == 0 and H <= 1024
    out = []
    t = t0
    while t < t0 + rows:
        n = min(t0 + rows - t, D - t % D)               # within one token block
        b = t // D
        for j in range(3):
            out.append(I.qst(src + (t - t0) * srs, vplane(vpl, H, b, j, D) + (t % D) // 4,
                             vsc + 4 * t, n, H // D, srs, 1, D // 4, row=True,
                             q6=j, q6t=True, ra=ra, comment=f"kv6 v b{b} plane {j}"))
        t += n
    return out


def pv(n, H, vpl, ones, out, ors, m, ab=0, acc=False, D=128):
    """O[m, H] = P' V^T's codes for tokens 0..n-1: P' (P[t] * S[t], zero past n to the block's
    end) QACT'ed in ACT rows 0..m-1, blocks ab + b for token block b; `ones` a 16-byte aligned
    DRAM plane of H fp32 1.0 (GRQ's scale plane). Tokens of a block past the written ones decode
    to q = -32: P' must be 0 there."""
    prog, first = [], not acc
    nb = -(-n // D)
    for j in range(3):
        prog += tmm_lut_plane(j)
        for b in range(nb):
            prog.append(I.tmm_cq(vplane(vpl, H, b, j, D), ones, out, H, 1, H * D // 4, ors, m,
                                 ab + b, 4 * H, 2, acc=not first, grq=True,
                                 comment=f"kv6 pv b{b} plane {j}"))
            first = False
    return prog


def tmm_lut_plane(j):
    return I.tmm_lut(LUTS[j], f"lutset kv6 plane {j}")


def dequant_v(dram, n, H, vpl, vsc, D=128):
    """The fp32 [n, H] V the planes stand for (q * S)."""
    V = np.zeros((n, H), np.float32)
    s = dram[vsc:vsc + 4 * n].copy().view(np.float32)
    for t in range(n):
        q = np.full(H, -32, np.int32)
        for j in range(3):
            a = vplane(vpl, H, t // D, j, D) + (t % D) // 4 + np.arange(H) * (D // 4)
            q += ((dram[a] >> (2 * (t % 4))) & 3).astype(np.int32) << (2 * j)
        V[t] = q * s[t]
    return V
