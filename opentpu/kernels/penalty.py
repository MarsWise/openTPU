"""The repetition penalty's factors over a prompt, on the device (no host arithmetic):

    pa[v] = 1 / R, pb[v] = R  for v among the context's ids;  1 elsewhere

(generate.Sampler's tables: the penalized logit is min(l * pa, l * pb)). The host writes only
the ids (input data, fp32 words) and their count; R and 1 / R come from the sampler's state
block (S_PEN, S_PENINV, written with the sampling settings). The program fills both tables
with 1.0 (VOP FILL into a TMEM buffer, then its stores), then a LOOP of the ids' count (RLD of
the count word: a run-time count) stores R and 1 / R at each id (RLD MUL id * 4 as the store's
address register). Repeated ids store the same values again (np.unique's effect).
"""
from __future__ import annotations

from .. import isa as I
from ..llm import generate as G


def program(cfg, *, ids: int, count: int, state: int, pa: int, pb: int, V: int,
            nmax: int, tmem: int = 0, buf: int = 4096) -> list:
    """The penalty-init program (one slice's: every slice holds the whole tables). ids: DRAM
    of nmax fp32 ids (the first `count` word's value used); count: DRAM byte address of the
    count word (fp32); state: the sampler's state block; pa, pb: the tables (V words). TMEM
    from `tmem`: a buffer of `buf` words, the state block, the ids."""
    if V <= 0 or nmax <= 0 or ids % 4 or count % 4 or pa % 4 or pb % 4 or state % 4:
        raise ValueError("penalty: word-aligned DRAM addresses, V and nmax > 0")
    buf = min(buf, V)
    t_buf, t_st = tmem, tmem + buf
    t_n = t_st + G.STATE_WORDS
    t_ids = t_n + 1
    if t_ids + nmax > cfg.TMEM_WORDS:
        raise ValueError("penalty: TMEM too small for the buffer and the ids")
    out = [I.vop(I.V_FILL, t_buf, 0, 0, 1, buf, 0, 0, 0, I.B_SCALAR, 1.0, comment="ones")]
    for k in range(0, V, buf):
        n = min(buf, V - k)
        out.append(I.st(pa + 4 * k, t_buf, n, comment="pa = 1"))
        out.append(I.st(pb + 4 * k, t_buf, n, comment="pb = 1"))
    out += [I.ld(state, t_st, G.STATE_WORDS, comment="R, 1 / R"),
            I.ld(count, t_n, 1, comment="the ids' count"),
            I.ld(ids, t_ids, nmax, comment="the ids"),
            I.li(1, 4, comment="bytes a word"),
            I.li(2, 0, comment="the id's index"),
            I.rld(3, t_n, bound=(0, nmax), comment="the count")]
    body = [I.rld(4, t_ids, ra=2, mul=0, rb=1, bound=(0, 4 * (V - 1)), comment="id * 4"),
            I.st(pa, t_st + G.S_PENINV, 1, ra=4, comment="pa[id] = 1 / R"),
            I.st(pb, t_st + G.S_PEN, 1, ra=4, comment="pb[id] = R"),
            I.addi(2, 2, 1, comment="the next id")]
    out.append(I.loop(len(body), 0, rcount=3, comment="per id"))
    out += body
    out.append(I.halt())
    if len(out) * 8 > cfg.IMEM_WORDS:
        raise ValueError("penalty: the program exceeds IMEM (a smaller V or a larger buf)")
    return out


def reference(V: int, context, penalty: float):
    """The tables the program writes (the host's former computation, for the tests)."""
    import numpy as np
    pa, pb = np.ones(V, np.float32), np.ones(V, np.float32)
    ix = np.unique(np.asarray(list(context), np.int64))
    pa[ix] = np.float32(1.0) / np.float32(penalty)
    pb[ix] = np.float32(penalty)
    return pa, pb
