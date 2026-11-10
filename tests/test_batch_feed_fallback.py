"""batch.Steps.feed: a feed program past TMEM (CompileError at compile, before any run) halves
the feed rows and feeds the same tokens; other compile errors propagate."""
from types import SimpleNamespace

import pytest

from opentpu.compiler import CompileError
from opentpu.llm import batch as BA


def steps(limit):
    st = BA.Steps.__new__(BA.Steps)
    st.eng = SimpleNamespace(cfg=SimpleNamespace(MCOLS=8), poss={0: 0}, cap=256)
    st.a = {"R": 8}
    st.feed_k, st.timing, runs = None, {}, []

    def go(grp, kinds, toks, uni, adv):
        if len(grp) > limit:
            raise CompileError(f"TMEM exhausted: a tile of {len(grp)} rows")
        assert all(k == BA.FEED for k in kinds) and [p for _, p in grp] == \
            list(range(grp[0][1], grp[0][1] + len(grp)))
        runs.append(list(toks))
        st.eng.poss[0] += len(grp)
    st._go = go
    return st, runs


def test_feed_halves_rows_after_tmem_failure():
    st, runs = steps(limit=4)
    toks = list(range(100, 123))
    st.feed(0, toks)
    assert sum(runs, []) == toks and max(map(len, runs)) == 4 and st.feed_rows() == 4
    assert st.eng.poss[0] == len(toks) and st.timing["feed_rows_cut"] == 4


def test_feed_other_compile_errors_propagate():
    st, _ = steps(limit=8)

    def go(*a):
        raise CompileError("IMEM: too long")
    st._go = go
    with pytest.raises(CompileError, match="IMEM"):
        st.feed(0, [1, 2, 3])
