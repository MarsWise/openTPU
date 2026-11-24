"""Maple-Preview (opentpu/llm/maple.py): a ternary MoE as lfm2's attention + MoE layers, its
experts streamed into DRAM slots. A tiny random Maple with TQ2_0's structure (d * {-1, 0, +1}
a block of 256) on the ISA simulator against the fp64 reference of Maple's own dimensions,
RoPE, window and clamped SwiGLU; the device's permuted head dimensions and RoPE tables are
checked against it, not assumed."""
import dataclasses

import numpy as np
import pytest

from opentpu import gguf as GG
from opentpu import quant as Q
from opentpu.isasim import board_config
from opentpu.llm import maple as M
from opentpu.llm.qwen3 import Engine

TOKS = [3, 17, 5, 42, 9, 260, 7, 11]


def _cos(a, b):
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


def _engine(spec, W, experts=None, wformat="tgsc", cap=256, **kw):
    cfg = board_config(DRAM_BYTES=1 << 27, MCOLS=8, TGSC=True)
    return Engine(spec.device(), M.DeviceWeights(W, spec), cap=cap, rows=1, cfg=cfg,
                  wformat=wformat, head_format="int8", experts=experts, **kw)


def _decode(eng, toks):
    return np.stack([np.asarray(eng.step(t), np.float64).reshape(-1) for t in toks])


@pytest.fixture(scope="module")
def tiny():
    spec = M.tiny_spec()
    return spec, M.tiny_weights(spec)


@pytest.fixture(scope="module")
def dev_full(tiny):
    spec, W = tiny
    eng = _engine(spec, W)
    return _decode(eng, TOKS), eng


def test_tq2_0_block_layout():
    """ggml's TQ2_0: qs[64] then fp16 d; element 32 (4 j + l) + m is bits 2l of qs[32 j + m],
    value (q - 1) d. One block with a distinct pattern: every element comes back where ggml
    puts it."""
    rng = np.random.default_rng(0)
    want = rng.integers(-1, 2, 256)
    qs = np.zeros(64, np.uint8)
    for e, v in enumerate(want):
        blk, m = divmod(e, 32)
        j, l = divmod(blk, 4)
        qs[32 * j + m] |= np.uint8((v + 1) << (2 * l))
    raw = np.concatenate([qs, np.frombuffer(np.float16(0.375).tobytes(), np.uint8)])
    got = M.tq2_values(raw, 1, 256)
    assert np.array_equal(got[0], want * 0.375)
    t, d = M.tq2_ternary(raw, 1, 256)
    assert np.array_equal(t[0], want) and d[0, 0] == 0.375
    bad = raw.copy()
    bad[0] |= 3
    with pytest.raises(ValueError, match="reserved"):
        M.tq2_values(bad, 1, 256)
    assert GG.BLOCK[GG.TQ2_0] == (256, 66)


def test_tgsc_packs_tq2_0_rows_losslessly(tiny):
    """A TQ2_0 matrix is d * {-1, 0, +1} in every 128-block: quant's TGSC rows (GSC's packing,
    no Hadamard signs) hold it exactly; a column part (an expert's W_down) is the same."""
    spec, W = tiny
    w = W["blk.0.attn_q.weight"]
    rows, sc = Q.quantize_mxu(w, Q.TGSC, 128)
    t = Q.unpack_ternary(rows[:, :w.shape[1] // 4], w.shape[1]).astype(np.float32)
    s = sc.view(np.float32).reshape(w.shape[0], -1)
    assert np.array_equal(np.abs(w).reshape(w.shape[0], -1, 128).max(-1), s)
    assert np.array_equal(t * np.repeat(s, 128, axis=1), w)
    wd = W.expert("blk.0.ffn_down_exps.weight", 3)
    q, s2 = Q.quantize_mxu(wd, Q.TGSC, 128)
    (pq, ps), = Q.column_parts(q, s2, Q.TGSC, 512, 512, 128)
    assert np.array_equal(pq, q) and np.array_equal(ps, s2)


def test_route_is_the_softmax_of_the_k_largest():
    rng = np.random.default_rng(1)
    lg = rng.normal(size=256)
    ids, w = M.route(lg, 8)
    top = np.argsort(-lg)[:8]
    e = np.exp(lg[top] - lg[top].max())
    assert list(ids) == list(top) and np.allclose(w, e / e.sum())


def test_device_follows_the_reference(tiny, dev_full):
    """ISA, TMM GSC ternary layers and experts (Maple's own values), int8 head and router:
    every row's logits within cos 0.98 of the fp64 reference (mean 0.995)."""
    spec, W = tiny
    dev, _ = dev_full
    c = _cos(dev, M.reference_logits(spec, W, TOKS))
    assert c.min() > 0.98 and c.mean() > 0.995, np.round(c, 4)


def test_rope_kinds_and_permutation_on_the_device():
    """Sharper attention (q / k gains x 3, two full layers): the device follows the reference,
    and is far from references with RoPE on every head dimension or on the full layers too
    (the tables and the head permutation are on the device, not dropped)."""
    spec = M.tiny_spec(full=(0, 2))
    W = M.tiny_weights(spec, seed=1, o_scale=0.12, qk_gain=3.0)
    dev = _decode(_engine(spec, W), TOKS)
    c = _cos(dev, M.reference_logits(spec, W, TOKS))
    assert c.min() > 0.97, np.round(c, 4)
    for wrong in (dataclasses.replace(spec, rot=128), dataclasses.replace(spec, full=())):
        assert _cos(dev, M.reference_logits(wrong, W, TOKS))[1:].min() < 0.9


def test_device_permutation_is_maples_rope():
    """PERM and theta^2: the device's lfm3-form tables on the permuted dimensions rotate q and
    k as Maple's rotate_half over its first 64 dimensions does (fp64, every position)."""
    spec = M.tiny_spec()
    rng = np.random.default_rng(2)
    v = rng.normal(size=(5, 128))
    c, s = spec.rope(np.arange(5))
    r = spec.rot
    a, b = v[:, :r // 2], v[:, r // 2:r]
    want = np.concatenate([a * c - b * s, b * c + a * s, v[:, r:]], axis=1)
    pv = v[:, M.PERM]
    dc, ds = spec.dev_tables(M.SLIDE, np.arange(5))
    h = 64
    got = np.concatenate([pv[:, :h] * dc - pv[:, h:] * ds, pv[:, h:] * dc + pv[:, :h] * ds], 1)
    assert np.allclose(got, want[:, M.PERM])
    fc, fs = spec.dev_tables(M.FULL, np.arange(5))
    assert np.all(fc == 1) and np.all(fs == 0)


def test_small_cache_is_bit_exact(tiny, dev_full):
    """k slots a layer (16 experts, 4 a token): most routes miss and the host streams the
    experts into slots; the logits do not move by a bit."""
    spec, W = tiny
    full, eng_full = dev_full
    eng = _engine(spec, W, experts=spec.k)
    small = _decode(eng, TOKS)
    assert np.array_equal(small, full)
    assert eng.server.misses > 0 and eng_full.server.misses == 0


def test_clamped_swiglu():
    """Experts whose gate and up values pass +-7 (weights x 8): the device clamps as
    modeling_maple does (silu(min(g, 7)) * clip(u, -7, 7)); a reference without the clamp is
    far from it."""
    spec = M.tiny_spec(layers=2, full=(1,))
    W = M.tiny_weights(spec, seed=4)
    for i in range(spec.layers):
        for n in ("ffn_gate_exps", "ffn_up_exps"):
            W[f"blk.{i}.{n}.weight"] = W[f"blk.{i}.{n}.weight"] * np.float32(8)
    toks = TOKS[:5]
    dev = _decode(_engine(spec, W), toks)
    ref = M.reference_logits(spec, W, toks)
    noc = M.reference_logits(dataclasses.replace(spec, clamp=1e9), W, toks)
    c = _cos(dev, ref)
    assert c.min() > 0.97, np.round(c, 4)
    assert _cos(dev, noc).min() < c.min() - 0.05


def test_refusals():
    spec = M.tiny_spec()
    W = M.tiny_weights(spec)
    with pytest.raises(NotImplementedError, match="sliding window"):
        _engine(spec, W, cap=1024)
    with pytest.raises(ValueError, match="clamp"):
        dataclasses.replace(spec, clamp=5.0).moe()


def test_spec_from_the_gguf_header():
    """Maple-Preview's header (maple-preview-TQ2_0-head-F16.gguf, read 2026-11-24): the spec
    and its checks."""
    kv = {"general.architecture": "maple", "maple.block_count": 24,
          "maple.embedding_length": 2048, "maple.attention.head_count": 16,
          "maple.attention.head_count_kv": 4, "maple.rope.freq_base": 10000.0,
          "maple.attention.layer_norm_rms_epsilon": 1e-6, "maple.expert_count": 256,
          "maple.expert_used_count": 8, "maple.attention.key_length": 128,
          "maple.attention.value_length": 128, "maple.vocab_size": 151936,
          "maple.rope.dimension_count": 64, "maple.attention.sliding_window": 512,
          "maple.attention.sliding_window_pattern": [True, True, True, False] * 6,
          "maple.expert_feed_forward_length": 512, "maple.expert_weights_scale": 1.0,
          "maple.expert_weights_norm": True, "maple.expert_gating_func": 1,
          "maple.swiglu_clamp_exp": [7.0] * 24, "tokenizer.ggml.eos_token_id": 151645,
          "tokenizer.ggml.bos_token_id": 151643}
    s = M.Spec.from_gguf(kv)
    assert s == M.Spec()
    assert s.full == (3, 7, 11, 15, 19, 23) and s.device().moe.act == "silu_clamp7"
    with pytest.raises(ValueError, match="clamps"):
        M.Spec.from_gguf({**kv, "maple.swiglu_clamp_exp": [7.0] * 23 + [6.0]})
    with pytest.raises(ValueError, match="maple"):
        M.Spec.from_gguf({**kv, "general.architecture": "llama"})


def _tq2(w: np.ndarray) -> bytes:
    """ggml's quantize of rows that are already d * {-1, 0, +1} a 256-block (TQ2_0)."""
    rows, K = w.shape
    b = w.reshape(rows, K // 256, 256)
    d = np.abs(b).max(-1).astype(np.float16)
    q = (np.sign(b) + 1).astype(np.uint8).reshape(rows, K // 256, 2, 4, 32)
    qs = sum(q[:, :, :, l, :] << (2 * l) for l in range(4)).astype(np.uint8).reshape(rows, -1, 64)
    return np.concatenate([qs, d[..., None].view(np.uint8).reshape(rows, -1, 2)], -1).tobytes()


def _write_gguf(path, W, kv):
    import struct

    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    hdr = b"GGUF" + struct.pack("<IQQ", 3, len(W), len(kv))
    for k, v in kv.items():
        hdr += s(k) + struct.pack("<I", 8) + s(v)
    data, infos = b"", b""
    for n, v in W.items():
        v = np.asarray(v, np.float32)
        if "exps" in n:
            typ, raw = GG.TQ2_0, b"".join(_tq2(v[e]) for e in range(v.shape[0]))
        elif n.endswith(("attn_q.weight", "attn_k.weight", "attn_v.weight",
                         "attn_output.weight")):
            typ, raw = GG.TQ2_0, _tq2(v)
        elif n in ("token_embd.weight", "output.weight"):
            typ, raw = GG.F16, v.astype(np.float16).tobytes()
        else:
            typ, raw = GG.F32, v.tobytes()
        infos += s(n) + struct.pack("<I", v.ndim) + b"".join(
            struct.pack("<Q", x) for x in reversed(v.shape)) + struct.pack("<IQ", typ, len(data))
        data += raw + b"\0" * ((-len(raw)) % 32)
    hdr += infos
    hdr += b"\0" * ((-len(hdr)) % 32)
    path.write_bytes(hdr + data)


def test_gguf_weights_read_a_maple_file(tmp_path):
    """A GGUF of the tiny Maple (TQ2_0 attention and stacked experts, F16 embedding and head,
    F32 rest): GGUFWeights gives the dict's values back (experts one at a time, embedding rows
    by index), and the device run on them equals the dict's bit for bit."""
    spec = M.tiny_spec(layers=2, full=(1,))
    W = M.tiny_weights(spec, seed=5)
    for n in ("token_embd.weight", "output.weight"):
        W[n] = W[n].astype(np.float16).astype(np.float32)
    f = tmp_path / "tiny.gguf"
    _write_gguf(f, W, {"general.architecture": "maple"})
    G = M.GGUFWeights(f)
    for n, v in W.items():
        if "exps" in n:
            assert all(np.array_equal(G.expert(n, e), v[e]) for e in (0, 7, 15)), n
        else:
            assert np.array_equal(G[n], v), n
    assert np.array_equal(G.rows("token_embd.weight", [4, 2]), W["token_embd.weight"][[4, 2]])
    toks = TOKS[:4]
    assert np.array_equal(_decode(_engine(spec, G, experts=4), toks),
                          _decode(_engine(spec, W, experts=4), toks))
    assert np.array_equal(M.reference_logits(spec, G, toks), M.reference_logits(spec, W, toks))
