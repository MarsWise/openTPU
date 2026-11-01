"""Sampled decoding with the device's own random numbers (VOP RAND, Config.VRAND): the host
writes a seed into the state block and no uniforms; position q's uniform is
rand.uniform(q, seed), drawn by the sampler on the device (generate.Sampler.uniform)."""
import dataclasses

import numpy as np
import pytest

from opentpu.llm import generate as G

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture(scope="module", params=["qwen3", "qwen35"])
def tiny(request):
    from test_autodecode import _tiny
    return (request.param,) + _tiny(request.param)


def _engines(spec, W, S, n=2, rng=True):
    from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config
    cfg = device_config(spec, 512, rows=PREFILL_ROWS, lookup=True, S=S)
    cfg = dataclasses.replace(cfg, VRAND=rng)
    return [Engine(spec, W, cap=512, cfg=cfg, resident=True) for _ in range(n)]


@pytest.mark.parametrize("S", [1, 2])
@pytest.mark.parametrize("split", [None, True])
def test_the_device_draws_the_uniforms(tiny, S, split):
    """12 sampled tokens across the bucket boundary: the reference picks with the device's
    uniforms (uniform_at(q, seed), the seed the host drew from the request's generator), and
    the host wrote no uniform (uni[] is untouched)."""
    name, W, spec = tiny
    if name != "qwen3" and S == 2:
        pytest.skip("one case for the other models")
    a, b = _engines(spec, W, S)
    a.gen_split = split
    samp = G.Sampling(0.8, 5, 0.9, 1.1)
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 248)]
    lg = a.prefill(toks)
    b.prefill(toks)
    ctx = toks + [int(np.argmax(lg))]
    uni = a.image.lookup["gen"]["uni"]
    before = [a.backend.read(s, uni, 4 * 513).copy() for s in range(S)]
    got = a.generate_card(ctx[-1], 12, stop_ids=[], sampling=samp, context=ctx,
                          rng=np.random.default_rng(7))
    seed = int(np.random.default_rng(7).integers(0, 1 << 32))
    assert a.last_seed == seed
    want, c = [], list(ctx)
    for i in range(12):
        q = len(c)                                  # the position the token lands at
        t = G.reference_pick(b.step(c[-1]), samp, c, float(G.uniform_at(q, seed)), S)
        want.append(t)
        c.append(t)
    assert got == want and len(set(got)) > 3
    for s in range(S):
        assert np.array_equal(a.backend.read(s, uni, 4 * 513), before[s])


def test_seeds_change_the_draws_and_repeat_them(tiny):
    """The same seed gives the same tokens (the counters are positions, not run order); a
    different seed different ones."""
    name, W, spec = tiny
    if name != "qwen3":
        pytest.skip("one model")
    runs = []
    for seed_src in (7, 7, 8):
        (a,) = _engines(spec, W, 1, n=1)
        toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 20)]
        ctx = toks + [int(np.argmax(a.prefill(toks)))]
        runs.append(a.generate_card(ctx[-1], 10, stop_ids=[], context=ctx,
                                    sampling=G.Sampling(1.5, 20, 1.0, 1.0),
                                    rng=np.random.default_rng(seed_src)))
    assert runs[0] == runs[1] and runs[0] != runs[2]


@pytest.mark.parametrize("S,split", [(1, None), (2, True)])
def test_device_rng_generate_on_rtl(have_verilator, tiny, S, split):
    """The sampled loop with VOP RAND on the Verilator RTL: from the same DRAM state, the
    ISA simulator's tokens and its whole DRAM, bit for bit."""
    from opentpu.llm.rtl_backend import RtlBackend
    name, W, spec = tiny
    if name != "qwen3":
        pytest.skip("one model")
    (eng,) = _engines(spec, W, S, n=1)
    eng.gen_split = split
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 248)]
    t0 = int(np.argmax(eng.prefill(toks)))
    samp = G.Sampling(0.8, 5, 0.9, 1.1)
    isa = eng.backend
    n = eng.image.nbytes
    rtl = RtlBackend(eng.cfg, [s.dram[:n] for s in isa.machine.slices])
    pos = eng.pos
    kw = dict(sampling=samp, context=toks + [t0])
    want = eng.generate_card(t0, 12, stop_ids=[], rng=np.random.default_rng(3), **kw)
    eng.backend, eng.pos, eng._chained = rtl, pos, {}
    got = eng.generate_card(t0, 12, stop_ids=[], rng=np.random.default_rng(3), **kw)
    assert got == want and len(set(got)) > 3
    for s in range(eng.cfg.S):
        assert np.array_equal(isa.machine.slices[s].dram[:n], rtl.drams[s][:n])


@pytest.mark.parametrize("span", [False, True])
def test_batched_rows_draw_the_decode_loops_uniforms(span):
    """Batched steps (llm/batch.py, the server's and vLLM's) on a VOP RAND device: a sampled
    row draws the uniform of the position its pick lands at under its slot's seed, the decode
    loop's draw there (uniform_at); the host passes no uniforms (they are ignored)."""
    from opentpu.llm import batch as BA
    from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config
    from test_serve_kvio import _model
    W, spec, cfg1 = _model("qwen3")
    cfg = device_config(spec, 512, batch=4, rows=PREFILL_ROWS, lookup=True, S=1)
    cfg = dataclasses.replace(cfg, DRAM_BYTES=2 * cfg.DRAM_BYTES, VRAND=True)
    eng = Engine(spec, W, cap=512, cfg=cfg, batch=4, resident=True, prompt_runs=True,
                 batch_area=True)
    st = BA.Steps(eng)
    if not span:
        st.span_steps = 0
    r = np.random.default_rng(1)
    slots, ps = [0, 2], [[int(x) for x in r.integers(0, 1000, n)] for n in (17, 25)]
    samps = {0: G.Sampling(0.8, 20, 0.9, 1.1), 2: G.Sampling(1.2, 8, 1.0, 1.0)}
    seeds = {0: 11, 2: 0xFFFFFFFF}
    refs, ctx, nxt = {}, {}, {}
    for s, p in zip(slots, ps):
        nxt[s] = int(np.argmax(eng.prefill(p, seq=s)))
        ctx[s] = list(p) + [nxt[s]]
        st.set_sampling(s, samps[s], ctx[s], seed=seeds[s])
        e = Engine(spec, W, cap=512, cfg=cfg1, resident=True, prompt_runs=True)
        e.prefill(p)
        refs[s] = e
    for _ in range(5):
        rows = [(s, eng.poss[s]) for s in slots]
        out = st.run(rows, [nxt[s] for s in slots], uniforms=[0.0, 0.0])
        for i, s in enumerate(slots):
            q = len(ctx[s])                         # the position the pick lands at
            want = G.reference_pick(refs[s].step(nxt[s]), samps[s], ctx[s],
                                    float(G.uniform_at(q, seeds[s])), 1, None)
            assert out[i] == want, (s, q)
            nxt[s] = out[i]
            ctx[s].append(out[i])
