"""LFM3 (opentpu/llm/lfm3.py): the spec from the checkpoint's config and the float reference.

The reference is checked against Hugging Face's LFM2 (transformers has no LFM3) on the part
both share -- a tiny model with LFM2's structure written in LFM3's names: no sandwich norms,
full RoPE, every gate sigmoid(0) = 1/2 (o_proj doubled), conv width 3 -- and LFM3's own parts
(the zero-tail partial RoPE, the sliding window, the per-head gate, the sandwich norms) on
their definitions."""
import json

import numpy as np
import pytest

from opentpu.llm import lfm3 as L3


def _tiny_spec(**kw):
    base = dict(hidden=128, kinds=(L3.CONV, L3.FULL, L3.CONV, L3.SLIDE), n_q=4, n_kv=2,
                head_dim=32, ffn=256, vocab=300, conv_k=7, window=4,
                rope={L3.FULL: (1e6, 0.5), L3.SLIDE: (1e4, 1.0)})
    base.update(kw)
    return L3.Spec(**base)


def _tiny_weights(spec, seed=0):
    r = np.random.default_rng(seed)
    H, d, P = spec.hidden, spec.head_dim, spec.prefix

    def lin(*s):
        return (0.2 * r.normal(size=s)).astype(np.float32)

    def g(n):
        return (1 + 0.1 * r.normal(size=n)).astype(np.float32)

    W = {P + "embed_tokens.weight": lin(spec.vocab, H), P + "embedding_norm.weight": g(H)}
    for i, k in enumerate(spec.kinds):
        p = f"{P}layers.{i}."
        W.update({p + "operator_norm.weight": g(H), p + "ffn_norm.weight": g(H),
                  p + "post_operator_norm.weight": g(H), p + "post_ffn_norm.weight": g(H),
                  p + "feed_forward.w1.weight": lin(spec.ffn, H),
                  p + "feed_forward.w3.weight": lin(spec.ffn, H),
                  p + "feed_forward.w2.weight": lin(H, spec.ffn)})
        if k == L3.CONV:
            W.update({p + "conv.in_proj.weight": lin(3 * H, H),
                      p + "conv.conv.weight": lin(H, 1, spec.conv_k),
                      p + "conv.out_proj.weight": lin(H, H)})
        else:
            a = p + "self_attn."
            W.update({a + "q_proj.weight": lin(spec.n_q * d, H),
                      a + "k_proj.weight": lin(spec.n_kv * d, H),
                      a + "v_proj.weight": lin(spec.n_kv * d, H),
                      a + "o_proj.weight": lin(H, spec.n_q * d), a + "q_norm.weight": g(d),
                      a + "k_norm.weight": g(d), a + "gate_proj.weight": lin(spec.n_q, H)})
    return W


def test_spec_from_the_omni_config(tmp_path):
    """LFM3-Omni's text_config: 24 layers (13 conv, 5 sliding, 6 full), conv 7, window 256,
    16 / 4 heads of 128, FFN 7680, full layers' RoPE theta 1e6 rotating half, the sliding
    layers' theta 1e4 whole; the decoder under model.language_model."""
    kinds = (["conv", "conv", "full_attention", "conv", "sliding_attention", "conv"] +
             ["full_attention", "conv", "sliding_attention", "conv"] * 4 + ["full_attention", "conv"])
    cfg = {"model_type": "lfm3_omni", "text_config": {
        "model_type": "lfm3", "hidden_size": 2048, "num_hidden_layers": 24, "layer_types": kinds,
        "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 128,
        "intermediate_size": 7680, "vocab_size": 128000, "conv_L_cache": 7, "sliding_window": 256,
        "norm_eps": 1e-5, "use_sandwich_norm": True, "tie_word_embeddings": True,
        "rope_parameters": {
            "full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 1e6, "rope_type": "default"},
            "sliding_attention": {"rope_theta": 10000, "rope_type": "default"}}}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    s = L3.Spec.from_hf(tmp_path)
    assert s.layers == 24 and s.kinds.count(L3.CONV) == 13
    assert s.kinds.count(L3.SLIDE) == 5 and s.kinds.count(L3.FULL) == 6
    assert (s.conv_k, s.window, s.n_q, s.n_kv, s.head_dim, s.ffn) == (7, 256, 16, 4, 128, 7680)
    assert s.prefix == "model.language_model."
    f = s.inv_freq(L3.FULL)
    assert np.count_nonzero(f) == 32 and f[0] == 1.0 and not f[32:].any()
    assert np.count_nonzero(s.inv_freq(L3.SLIDE)) == 64
    c, sn = s.rope_tables(L3.FULL, 5)                 # dims 32..63 and 96..127 unrotated
    assert np.all(c[32:64] == 1) and np.all(sn[32:64] == 0) and np.all(c[96:] == 1)
    assert not np.all(c[:32] == 1) and not np.all(c[64:96] == 1)


def test_reference_matches_hf_lfm2_on_the_shared_structure():
    """LFM2's structure in LFM3's names (no sandwich, full RoPE, gates 1/2 with o_proj
    doubled, conv 3) == transformers Lfm2ForCausalLM."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    kinds = ("conv", "attn", "conv", "attn")
    hc = transformers.Lfm2Config(
        hidden_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=256, vocab_size=300, norm_eps=1e-5,
        layer_types=["full_attention" if k == "attn" else "conv" for k in kinds],
        conv_L_cache=3, conv_bias=False, block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True, max_position_embeddings=512, initializer_range=0.2,
        rope_parameters={"rope_type": "default", "rope_theta": 1e6})
    m = transformers.Lfm2ForCausalLM(hc).float().eval()
    sd = {k: v.detach().float().numpy() for k, v in m.state_dict().items()}
    spec = L3.Spec(hidden=128, kinds=tuple(L3.CONV if k == "conv" else L3.FULL for k in kinds),
                   n_q=4, n_kv=2, head_dim=32, ffn=256, vocab=300, conv_k=3, window=1 << 30,
                   rope={L3.FULL: (1e6, 1.0)}, sandwich=False)
    W = {}
    for k, v in sd.items():
        n = (k.replace("self_attn.q_layernorm", "self_attn.q_norm")
              .replace("self_attn.k_layernorm", "self_attn.k_norm")
              .replace("self_attn.out_proj", "self_attn.o_proj"))
        if n.endswith("self_attn.o_proj.weight"):
            v = v * 2.0                                   # the gate sigmoid(0) = 1/2
        W[n] = v
    for i, k in enumerate(kinds):
        if k == "attn":
            W[f"model.layers.{i}.self_attn.gate_proj.weight"] = np.zeros((4, 128), np.float32)
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 300, 9)]
    with torch.no_grad():
        want = m(torch.tensor([toks])).logits[0].numpy()
    got = L3.reference_logits(spec, W, toks)
    np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)


def test_lfm3_parts_by_definition():
    """The sliding window (a token farther back than the window weighs nothing), the zero-tail
    RoPE (unrotated pairs keep position-independent scores) and the sandwich norms / gates
    change the logits; the window 1 << 30 model equals the full one when T <= window."""
    spec = _tiny_spec()
    W = _tiny_weights(spec)
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 300, 10)]
    a = L3.reference_logits(spec, W, toks)
    assert np.isfinite(a).all()
    # window 4: changing a token 5 back changes the sliding layer's view -- but the full layer
    # still sees it; with every attention layer sliding, a token outside the window and
    # outside the conv's reach (7 taps) leaves the last logits unchanged only through no path:
    # so check the window directly on a one-layer sliding model with no conv
    one = _tiny_spec(kinds=(L3.SLIDE,), window=3)
    W1 = _tiny_weights(one, seed=2)
    t = [5, 6, 7, 8, 9, 10]
    t2 = [99] + t[1:]                                  # position 0 is outside the last window
    x1 = L3.reference_logits(one, W1, t)
    x2 = L3.reference_logits(one, W1, t2)
    np.testing.assert_array_equal(x1[-1], x2[-1])      # the last query sees positions 3..5
    assert not np.array_equal(x1[0], x2[0])
    wide = L3.reference_logits(_tiny_spec(kinds=(L3.SLIDE,), window=1 << 30), W1, t)
    assert not np.allclose(wide[-1], x1[-1])
    # no sandwich: different logits (the norms are not the identity)
    b = L3.reference_logits(_tiny_spec(sandwich=False), W, toks)
    assert not np.allclose(a, b)


# ----------------------------------------------------------------------------- device graph
import dataclasses  # noqa: E402

from opentpu.isasim import board_config  # noqa: E402
from opentpu.llm.qwen3 import Engine  # noqa: E402


def _cos_rows(a, b):
    return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


def _engine(spec, W, cap=256, **kw):
    return Engine(L3.device_spec(spec), L3.DeviceWeights(W, spec.prefix), cap=cap, rows=1,
                  cfg=kw.pop("cfg", board_config(DRAM_BYTES=1 << 26)), **kw)


def _decode(eng, toks):
    return np.stack([np.asarray(eng.step(t)).reshape(-1) for t in toks])


def test_device_graph_follows_the_reference():
    """The device graph (ISA, W8A8, int8 KV) against the fp32 reference: conv layers of 7 taps,
    a full and a sliding layer (window 4: the window inside one block, its start masked), head
    gates and sandwich norms. The same logits miss a reference without the window or the gates
    by far (the parts are on the device, not dropped)."""
    spec = _tiny_spec()
    W = _tiny_weights(spec)
    toks = list(range(3, 15))
    dev = _decode(_engine(spec, W), toks)
    ref = L3.reference_logits(spec, W, toks)
    assert _cos_rows(dev, ref).min() > 0.995
    nowin = L3.reference_logits(dataclasses.replace(spec, window=10 ** 6), W, toks)
    assert _cos_rows(dev, nowin).min() < 0.95
    W2 = {k: (np.zeros_like(v) if k.endswith("gate_proj.weight") else v) for k, v in W.items()}
    assert _cos_rows(dev, L3.reference_logits(spec, W2, toks)).min() < 0.9


def test_sliding_window_across_blocks():
    """window 256 = one attention block: past position 255 the window spans two blocks, the
    first masked before its start (attention.Blocks); 300 tokens."""
    spec = _tiny_spec(window=256, kinds=(L3.CONV, L3.SLIDE, L3.CONV, L3.FULL))
    W = _tiny_weights(spec, 3)
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 300, 300)]
    dev = _decode(_engine(spec, W, cap=512), toks)
    ref = L3.reference_logits(spec, W, toks)
    nowin = L3.reference_logits(dataclasses.replace(spec, window=10 ** 6), W, toks)
    assert _cos_rows(dev, ref).min() > 0.99
    assert _cos_rows(dev[280:], nowin[280:]).mean() < 0.985     # (the window drops tokens)
    assert _cos_rows(dev[280:], ref[280:]).mean() > 0.995


def test_device_graph_refuses_what_it_lacks():
    spec = _tiny_spec()
    W = _tiny_weights(spec)
    with pytest.raises(NotImplementedError):
        Engine(L3.device_spec(spec), L3.DeviceWeights(W, spec.prefix), cap=256,
               cfg=board_config(DRAM_BYTES=1 << 26))          # chunked prefill (rows > 1)
    with pytest.raises(ValueError, match="sandwich"):
        L3.device_spec(dataclasses.replace(spec, sandwich=False))


def test_tiny_lfm3_on_board_model(have_verilator):
    """The board model (Verilator, r8's WAITC sequencer) through the host driver: logits bit
    for bit the ISA simulator's, through a full turn of the 7-tap conv ring and the sliding
    window's masked start."""
    from opentpu.host.board import BoardBackend, SimTransport
    spec = _tiny_spec()
    W = _tiny_weights(spec)
    cfg = board_config(DRAM_BYTES=1 << 23, SEQ_WC=True)
    isa = _engine(spec, W, cfg=cfg)
    tr = SimTransport(ch_bytes=cfg.DRAM_BYTES // 2, stall=20, seed=5,
                      params={"SEQ_WC": 1, "WIN": 16})
    brd = _engine(spec, W, cfg=cfg, backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    for tok in (11, 22, 33, 44, 55, 66, 77, 88, 99):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(np.asarray(a).view(np.uint32), np.asarray(b).view(np.uint32))


def test_resident_decode_and_generate_on_the_card():
    """Run-time positions (resident decode: the token id and position as run arguments, the
    embedding gathered on the device) bit for bit the per-position programs, through the
    sliding window's two-block form (window = one block, positions past 256); then the decode
    loop on the device (generate_card: the token picked on the card and fed back) the host's
    greedy loop's tokens."""
    # (head_dim 128 as LFM3's: heads narrower than D, padded, differ in the last bits between
    # the two forms past the first block -- qwen3's attention, not this file's)
    spec = _tiny_spec(window=256, kinds=(L3.CONV, L3.SLIDE, L3.CONV, L3.FULL), head_dim=128)
    W = _tiny_weights(spec, 3)
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 300, 270)]
    a = _engine(spec, W, cap=512)
    b = _engine(spec, W, cap=512, resident=True)
    assert b.resident
    for t in toks:
        x, y = a.step(t), b.step(t)
        assert np.array_equal(np.asarray(x).view(np.uint32), np.asarray(y).view(np.uint32))
    want, cur = [], toks[-1]
    for _ in range(4):
        cur = int(np.argmax(a.step(cur)))
        want.append(cur)
    assert b.generate_card(toks[-1], 4, stop_ids=[]) == want


def test_real_weights_follow_the_reference():
    """LFM3-Omni's text decoder (24 layers, real bf16 weights extracted by row range:
    OTPU_LFM3_DIR, PartWeights) on the device graph: "The capital of France is" -> every
    token's logits cos >= 0.9995 against the fp32 reference, the same next token (" Paris")."""
    import os
    d = os.environ.get("OTPU_LFM3_DIR", "/home/claude/lfm3w")
    if not os.path.exists(os.path.join(d, "index.json")):
        pytest.skip("no extracted LFM3 weights (OTPU_LFM3_DIR)")
    tk = pytest.importorskip("tokenizers")
    W = L3.PartWeights(d)
    spec = L3.truncated(L3.Spec.from_hf(d), W.layers())
    ids = [spec.bos] + tk.Tokenizer.from_file(os.path.join(d, "tokenizer.json")).encode(
        "The capital of France is", add_special_tokens=False).ids
    ref = L3.reference_logits(spec, W, ids)
    dev = _decode(_engine(spec, W, cap=256, cfg=board_config(DRAM_BYTES=1 << 31)), ids)
    assert _cos_rows(dev, ref).min() > 0.9995
    assert int(np.argmax(dev[-1])) == int(np.argmax(ref[-1]))
