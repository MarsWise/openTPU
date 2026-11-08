"""The 6-bit K cache in the LLM attention graph (Image kv="kv6", compiler.KVDesc kfmt kv6):
K appended as QST Q6's three 2-bit planes a D-block, q.K^T as three TMM CQ2 GRQs a score block
(kernels/kv6.py's layout and LUTs), V int8 as before. A tiny random Qwen3 (numpy weights, no
torch) on the ISA simulator against the float64 emulation of the same 6-bit K; the resident
decode, the prefill and the on-card generate loop against the per-position programs."""
import numpy as np
import pytest

from opentpu import isa as I
from opentpu.compiler import CompileError, KVDesc, kv6_k_bytes
from opentpu.llm.qwen3 import Engine, Spec, device_config, emulated_logits

SPEC = Spec(256, 2, 4, 2, 128, 512, 1000)
R8 = dict(QST6=True, TCQ=True, TGRQ=True, TGSC=True)


def _weights(seed=0, H=256, L=2, nq=4, nkv=2, d=128, F_=512, V=1000):
    """Random Qwen3 weights in the checkpoint's names (HF's initializer_range 0.2: the greedy
    tokens vary), norms 1 + 0.1 N."""
    r = np.random.default_rng(seed)

    def lin(*s):
        return (0.2 * r.normal(size=s)).astype(np.float32)

    def g(n):
        return (1 + 0.1 * r.normal(size=n)).astype(np.float32)

    W = {"model.embed_tokens.weight": lin(V, H), "model.norm.weight": g(H)}
    for i in range(L):
        p = f"model.layers.{i}."
        W.update({p + "input_layernorm.weight": g(H), p + "post_attention_layernorm.weight": g(H),
                  p + "self_attn.q_norm.weight": g(d), p + "self_attn.k_norm.weight": g(d),
                  p + "self_attn.q_proj.weight": lin(nq * d, H),
                  p + "self_attn.k_proj.weight": lin(nkv * d, H),
                  p + "self_attn.v_proj.weight": lin(nkv * d, H),
                  p + "self_attn.o_proj.weight": lin(H, nq * d),
                  p + "mlp.gate_proj.weight": lin(F_, H), p + "mlp.up_proj.weight": lin(F_, H),
                  p + "mlp.down_proj.weight": lin(H, F_)})
    return W


@pytest.fixture(scope="module")
def W():
    return _weights()


def _cos(a, b):
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


def _engine(W, kv, cap=256, monkeypatch=None, **kw):
    monkeypatch.setenv("OTPU_KV", kv)
    lookup = kw.get("resident", False)
    return Engine(SPEC, W, cap=cap, cfg=device_config(SPEC, cap, lookup=lookup, **R8), **kw)


def test_kv6_decode_follows_the_6bit_emulation(W, monkeypatch):
    """Decode token by token (positions 0..9: score blocks of 1..10 tokens, n % 4 != 0 rounded
    up and masked): the logits follow the emulation of the same 6-bit K, the greedy tokens
    equal; K is stored by QST Q6 and read by TMM CQ2 GRQ only, its image smaller than int8's."""
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 10)]
    eng = _engine(W, "kv6", monkeypatch=monkeypatch)
    assert eng.image.kv == "kv6"
    dev = np.array([eng.step(t) for t in toks])
    emu = emulated_logits(SPEC, W, toks, kv="kv6")
    assert _cos(dev, emu).min() > 0.9995
    assert (dev.argmax(-1) == emu.argmax(-1)).all()
    prog = eng.backend.machine.slices[0].prog
    tmm = [p for p in prog if p.op == I.TMM and p.w[3] & I.TMM_GRQ]
    assert len(tmm) == 3 and all(p.w[3] & I.TMM_CQ for p in tmm)
    q6 = [p for p in prog if p.op == I.QST and "kv6 K store" in p.comment]
    assert len(q6) == 3                                   # one KV head a slice, three planes
    i8 = _engine(W, "int8", monkeypatch=monkeypatch)
    i8dev = np.array([i8.step(t) for t in toks])
    assert _cos(dev, i8dev).min() > 0.995                 # 6-bit K: near int8 K
    # K: 6 bits + an fp32 S a 128-block, against 8 bits + a row scale
    assert eng.image.kv_bytes < i8.image.kv_bytes
    cap, d = 256, 128
    assert sum(kv6_k_bytes(cap, d, 128)) == cap * (3 * d // 4 + 4)


def test_kv6_resident_prefill_and_buckets_are_bit_exact(W, monkeypatch):
    """The resident decode (one program per attention bucket, masked last block) and a chunked
    prefill give the per-position programs' logits bit for bit, across the bucket boundary 256,
    and both follow the emulation."""
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 262)]
    a = _engine(W, "kv6", cap=512, monkeypatch=monkeypatch, resident=True)
    b = _engine(W, "kv6", cap=512, monkeypatch=monkeypatch)
    la, lb = [], []
    for t in toks[:3]:
        la.append(a.step(t))
        lb.append(b.step(t))
    pa, pb = a.prefill(toks[3:252]), b.prefill(toks[3:252])
    assert np.array_equal(np.asarray(pa).view(np.uint32), np.asarray(pb).view(np.uint32))
    for t in toks[252:]:
        la.append(a.step(t))
        lb.append(b.step(t))
    la, lb = np.array(la), np.array(lb)
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    assert sorted(a._decodes) == [1, 2]
    emu = emulated_logits(SPEC, W, toks, kv="kv6")
    sel = np.concatenate([emu[:3], emu[252:]])
    assert _cos(la, sel).min() > 0.999
    assert (la.argmax(-1) == sel.argmax(-1)).all()


def test_kv6_generate_card_equals_host_steps(W, monkeypatch):
    """The on-card generate loop (tokens picked and fed back on the device) over a 6-bit K
    gives the tokens of host-driven steps."""
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 21)]
    e = _engine(W, "kv6", cap=512, monkeypatch=monkeypatch, resident=True)
    e.prefill(toks[:-1])
    card = e.generate_card(toks[-1], 12, stop_ids=())
    f = _engine(W, "kv6", cap=512, monkeypatch=monkeypatch, resident=True)
    f.prefill(toks[:-1])
    host, t = [], toks[-1]
    for _ in range(12):
        t = int(np.argmax(f.step(t)))
        host.append(t)
    assert card == host
    assert len(set(host)) > 1                            # the tokens vary


def test_kv6_refusals(W, monkeypatch):
    """kv6 needs r8's QST6 + TCQ + TGRQ; its K is not an int8 K (KVDesc.k refuses); saving a
    context of it (serve/kvio) is not implemented and says so."""
    monkeypatch.setenv("OTPU_KV", "kv6")
    with pytest.raises(ValueError, match="QST6"):
        SPEC.image(device_config(SPEC, 256, TCQ=True, TGRQ=True, TGSC=True), 256)
    with pytest.raises(ValueError, match="int8, kv6"):
        SPEC.image(device_config(SPEC, 256, **R8), 256, kv="kv5")
    kv = KVDesc({0: {"kp": 0, "ks": 0, "vt": 0, "vs": 0}}, 256, 128, 128, 1, 0, kfmt="kv6")
    with pytest.raises(CompileError, match="kv6"):
        kv.k(0)
    assert len(kv.k6(0)) == 3 and all(q.grq and q.cq == 2 for q in kv.k6(0))
    with pytest.raises(CompileError, match="6-bit K cache"):
        KVDesc({}, 254, 128, 128, 1, 0, kfmt="kv6")
    from opentpu.serve import kvio
    eng = _engine(W, "kv6", monkeypatch=monkeypatch)
    with pytest.raises(NotImplementedError, match="kv6"):
        kvio.Layout(eng.image, eng.cfg.S)


def test_kv6_token_on_board_rtl_is_bit_exact(have_verilator, W, monkeypatch):
    """A resident token at position 261 (score blocks of 256 and a masked bucket block) over a
    6-bit K on the Verilator RTL (the board's memory path, m8 with r8's QST6 + TCQ + TGRQ)
    equals the ISA simulator bit for bit: the logits and the whole DRAM image (the K planes and
    S planes the token appended)."""
    from opentpu import rtlsim
    from opentpu.isasim import board_config
    from opentpu.llm.rtl_backend import RtlBackend
    monkeypatch.setenv("OTPU_KV", "kv6")
    cfg = board_config(DRAM_BYTES=1 << 26, MCOLS=8, **R8)
    toks = [int(t) for t in np.random.default_rng(5).integers(0, 1000, 262)]
    eng = Engine(SPEC, W, cap=512, cfg=cfg, resident=True)
    assert eng.image.kv == "kv6"
    eng.prefill(toks[:-1])
    n, isa = eng.image.nbytes, eng.backend
    rtl = RtlBackend(eng.cfg, [s.dram[:n] for s in isa.machine.slices],
                     uarch=rtlsim.BOARD_UARCH, axi=True, boot=True)
    want = eng.step(toks[-1])
    eng.backend, eng.pos = rtl, eng.pos - 1
    got = eng.step(toks[-1])
    assert np.array_equal(want.view(np.uint32), got.view(np.uint32))
    assert np.array_equal(isa.machine.slices[0].dram[:n], rtl.drams[0][:n])


# ---- 6-bit K and V (Image kv="k6v6"): V^T as QST Q6T's planes, a group of four tokens a byte
# written from its first token (the group's fp32 rows in DRAM), P'.V as three TMM CQ2 GRQs

def test_k6v6_decode_follows_the_6bit_emulation(W, monkeypatch):
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 10)]
    eng = _engine(W, "k6v6", monkeypatch=monkeypatch)
    assert eng.image.kv == "k6v6"
    dev = np.array([eng.step(t) for t in toks])
    emu = emulated_logits(SPEC, W, toks, kv="k6v6")
    assert _cos(dev, emu).min() > 0.9995
    assert (dev.argmax(-1) == emu.argmax(-1)).all()
    prog = eng.backend.machine.slices[0].prog
    assert len([p for p in prog if p.op == I.QST and "kv6 V store" in p.comment]) == 3
    assert len([p for p in prog if p.op == I.TMM and p.w[3] & I.TMM_GRQ]) == 6
    k6 = _engine(W, "kv6", monkeypatch=monkeypatch)
    assert eng.image.kv_bytes < k6.image.kv_bytes


def test_k6v6_resident_prefill_and_generate_are_bit_exact(W, monkeypatch):
    """The resident decode (the group's slot, first token and byte as run arguments), a
    chunked prefill (groups the chunk starts inside go through the group's rows) and the
    on-card generate loop (the group words stepped on the device, across the bucket boundary
    256) against the per-position programs / host steps."""
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 262)]
    a = _engine(W, "k6v6", cap=512, monkeypatch=monkeypatch, resident=True)
    b = _engine(W, "k6v6", cap=512, monkeypatch=monkeypatch)
    la, lb = [], []
    for t in toks[:3]:
        la.append(a.step(t))
        lb.append(b.step(t))
    pa, pb = a.prefill(toks[3:252]), b.prefill(toks[3:252])
    assert np.array_equal(np.asarray(pa).view(np.uint32), np.asarray(pb).view(np.uint32))
    for t in toks[252:]:
        la.append(a.step(t))
        lb.append(b.step(t))
    la, lb = np.array(la), np.array(lb)
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    emu = emulated_logits(SPEC, W, toks, kv="k6v6")
    sel = np.concatenate([emu[:3], emu[252:]])
    assert _cos(la, sel).min() > 0.999 and (la.argmax(-1) == sel.argmax(-1)).all()
    for pre in (20, 250):
        e = _engine(W, "k6v6", cap=512, monkeypatch=monkeypatch, resident=True)
        e.prefill(toks[:pre - 1])
        card = e.generate_card(toks[pre - 1], 12, stop_ids=())
        f = _engine(W, "k6v6", cap=512, monkeypatch=monkeypatch, resident=True)
        f.prefill(toks[:pre - 1])
        host, t = [], toks[pre - 1]
        for _ in range(12):
            t = int(np.argmax(f.step(t)))
            host.append(t)
        assert card == host


def test_k6v6_refusals(W, monkeypatch):
    """A 6-bit V needs head dim 128 (the run-time group byte's stride) and has no int8 V^T."""
    from dataclasses import replace
    monkeypatch.setenv("OTPU_KV", "k6v6")
    with pytest.raises(ValueError, match="head dim 128"):
        replace(SPEC, head_dim=256).image(device_config(replace(SPEC, head_dim=256), 256,
                                                        **R8), 256, kv="k6v6")
    kv = KVDesc({0: {"kp": 0, "ks": 0, "vp": 0, "vg": 0, "vs": 0}}, 256, 128, 128, 1, 0,
                kfmt="kv6", vfmt="kv6", vones=4096, vplanes=2)
    with pytest.raises(CompileError, match="kv6"):
        kv.vt(0)
    assert len(kv.v6(0, 0, 256)) == 3
    with pytest.raises(CompileError, match="D-blocks"):
        kv.v6(0, 0, 384)


def test_k6v6_token_on_board_rtl_is_bit_exact(have_verilator, W, monkeypatch):
    """A resident token at position 261 over 6-bit K and V on the board RTL == ISA: the logits
    and the whole DRAM image (the K and V planes, the group's rows)."""
    from opentpu import rtlsim
    from opentpu.isasim import board_config
    from opentpu.llm.rtl_backend import RtlBackend
    monkeypatch.setenv("OTPU_KV", "k6v6")
    cfg = board_config(DRAM_BYTES=1 << 26, MCOLS=8, **R8)
    toks = [int(t) for t in np.random.default_rng(5).integers(0, 1000, 262)]
    eng = Engine(SPEC, W, cap=512, cfg=cfg, resident=True)
    eng.prefill(toks[:-1])
    n, isa = eng.image.nbytes, eng.backend
    rtl = RtlBackend(eng.cfg, [s.dram[:n] for s in isa.machine.slices],
                     uarch=rtlsim.BOARD_UARCH, axi=True, boot=True)
    want = eng.step(toks[-1])
    eng.backend, eng.pos = rtl, eng.pos - 1
    got = eng.step(toks[-1])
    assert np.array_equal(want.view(np.uint32), got.view(np.uint32))
    assert np.array_equal(isa.machine.slices[0].dram[:n], rtl.drams[0][:n])
