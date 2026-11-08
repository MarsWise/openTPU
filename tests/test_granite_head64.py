"""A small Llama-like model with Granite's 5:1 GQA and 64-wide heads."""
import numpy as np
import pytest

from opentpu.isasim import Config
from opentpu.llm.qwen3 import Engine, Spec, reference_logits


def weights(seed=18, n_q=10, n_kv=2, head_dim=64):
    rng = np.random.default_rng(seed)
    def mat(n, k):
        return rng.normal(0, 0.04, (n, k)).astype(np.float32)
    w = {"model.embed_tokens.weight": mat(256, 256), "model.norm.weight": np.ones(256, np.float32)}
    p = "model.layers.0."
    w[p + "input_layernorm.weight"] = np.ones(256, np.float32)
    w[p + "post_attention_layernorm.weight"] = np.ones(256, np.float32)
    for name, n, k in (("self_attn.q_proj.weight", n_q * head_dim, 256),
                       ("self_attn.k_proj.weight", n_kv * head_dim, 256),
                       ("self_attn.v_proj.weight", n_kv * head_dim, 256),
                       ("self_attn.o_proj.weight", 256, n_q * head_dim),
                       ("mlp.gate_proj.weight", 512, 256),
                       ("mlp.up_proj.weight", 512, 256),
                       ("mlp.down_proj.weight", 256, 512)):
        w[p + name] = mat(n, k)
    return w


@pytest.mark.parametrize("mcols", [4, 8])
@pytest.mark.parametrize("qk_norm", [False, True])
@pytest.mark.parametrize("slices", [1, 2])
def test_64_wide_heads_gqa5_on_128_wide_isa(monkeypatch, mcols, qk_norm, slices):
    for key in ("OTPU_IMG_CACHE", "OTPU_IMAGE_CACHE", "OTPU_PROG_CACHE"):
        monkeypatch.setenv(key, "0")
    spec = Spec(hidden=256, layers=1, n_q=10, n_kv=2, head_dim=64,
                ffn=512, vocab=256, qk_norm=qk_norm, tied=True)
    cfg = Config(S=slices, D=128, MCOLS=mcols, ACT_ROWS=8, DRAM_BYTES=32 << 20)
    w = weights()
    if qk_norm:
        w["model.layers.0.self_attn.q_norm.weight"] = np.ones(64, np.float32)
        w["model.layers.0.self_attn.k_norm.weight"] = np.ones(64, np.float32)
    tokens = [3, 17, 42, 51]
    reference = reference_logits(spec, w, tokens)
    engine = Engine(spec, w, cap=128, cfg=cfg)
    actual = np.stack([engine.step(token) for token in tokens])
    cosine = (actual * reference).sum(1) / np.linalg.norm(actual, axis=1) / np.linalg.norm(reference, axis=1)
    assert cosine.min() > 0.99


def test_64_wide_kv_uses_padded_non_overlapping_regions():
    spec = Spec(hidden=256, layers=1, n_q=10, n_kv=2, head_dim=64,
                ffn=512, vocab=256, qk_norm=False, tied=True)
    cfg = Config(D=128, MCOLS=8, ACT_ROWS=8, DRAM_BYTES=32 << 20)
    image = spec.image(cfg, cap=128, lookup=True)
    assert image.dk == 128
    kv = image.lofs["kvs"][0]
    for h in kv:
        assert h["ks"] >= h["k"] + 128 * 128
        assert h["vs"] >= h["vt"] + 128 * 128
    assert kv[1]["k"] >= kv[0]["vs"] + 128 * 4


@pytest.mark.parametrize("head_dim", [0, 63, 192])
def test_invalid_head_geometry_rejected(head_dim):
    spec = Spec(hidden=256, layers=1, n_q=10, n_kv=2, head_dim=head_dim,
                ffn=512, vocab=256, qk_norm=False, tied=True)
    with pytest.raises(ValueError, match="head_dim"):
        spec.check(Config(D=128, MCOLS=8))


@pytest.mark.parametrize("mcols", [4, 8])
def test_64_wide_prefill_matches_framework(monkeypatch, mcols):
    for key in ("OTPU_IMG_CACHE", "OTPU_IMAGE_CACHE", "OTPU_PROG_CACHE"):
        monkeypatch.setenv(key, "0")
    spec = Spec(hidden=256, layers=1, n_q=10, n_kv=2, head_dim=64,
                ffn=512, vocab=256, qk_norm=False, tied=True)
    cfg = Config(D=128, MCOLS=mcols, ACT_ROWS=8, DRAM_BYTES=32 << 20)
    w = weights()
    tokens = [3, 17, 42, 51]
    reference = reference_logits(spec, w, tokens)[-1]
    engine = Engine(spec, w, cap=128, cfg=cfg, resident=True, rows=2, pipeline=False)
    actual = engine.prefill(tokens)
    cosine = np.dot(actual, reference) / np.linalg.norm(actual) / np.linalg.norm(reference)
    assert cosine > 0.99


@pytest.mark.parametrize("mcols", [4, 8])
def test_128_wide_gqa8_prefill_and_decode_isa(monkeypatch, mcols):
    """MiniCPM-style 8:1 grouping spans two MMs on m4."""
    for key in ("OTPU_IMG_CACHE", "OTPU_IMAGE_CACHE", "OTPU_PROG_CACHE"):
        monkeypatch.setenv(key, "0")
    spec = Spec(hidden=256, layers=1, n_q=16, n_kv=2, head_dim=128,
                ffn=512, vocab=256, qk_norm=False, tied=True)
    cfg = Config(D=128, MCOLS=mcols, ACT_ROWS=8, DRAM_BYTES=32 << 20)
    w = weights(n_q=16, n_kv=2, head_dim=128)
    tokens = [3, 17, 42, 51]
    reference = reference_logits(spec, w, tokens)
    engine = Engine(spec, w, cap=128, cfg=cfg, resident=True, rows=2, pipeline=False)
    prefill = engine.prefill(tokens[:3])
    decoded = engine.step(tokens[3])
    for actual, ref in ((prefill, reference[2]), (decoded, reference[3])):
        cosine = np.dot(actual, ref) / np.linalg.norm(actual) / np.linalg.norm(ref)
        assert cosine > 0.99


def test_64_wide_gqa5_prefill_crosses_128_token_boundary(monkeypatch):
    for key in ("OTPU_IMG_CACHE", "OTPU_IMAGE_CACHE", "OTPU_PROG_CACHE"):
        monkeypatch.setenv(key, "0")
    spec = Spec(hidden=256, layers=1, n_q=10, n_kv=2, head_dim=64,
                ffn=512, vocab=256, qk_norm=False, tied=True)
    cfg = Config(D=128, MCOLS=4, ACT_ROWS=8, DRAM_BYTES=32 << 20)
    w = weights()
    tokens = np.random.default_rng(44).integers(0, spec.vocab, size=130).tolist()
    reference = reference_logits(spec, w, tokens)
    engine = Engine(spec, w, cap=256, cfg=cfg, resident=True, rows=2, pipeline=False)
    prefill = engine.prefill(tokens[:129])
    decoded = engine.step(tokens[129])
    for actual, ref in ((prefill, reference[128]), (decoded, reference[129])):
        cosine = np.dot(actual, ref) / np.linalg.norm(actual) / np.linalg.norm(ref)
        assert cosine > 0.99


@pytest.mark.parametrize("mcols", [4, 8])
def test_64_wide_gqa5_decode_rtl_matches_isa(monkeypatch, have_verilator, mcols):
    from opentpu.llm.rtl_backend import RtlBackend
    for key in ("OTPU_IMG_CACHE", "OTPU_IMAGE_CACHE", "OTPU_PROG_CACHE"):
        monkeypatch.setenv(key, "0")
    spec = Spec(hidden=256, layers=1, n_q=10, n_kv=2, head_dim=64,
                ffn=512, vocab=256, qk_norm=False, tied=True)
    cfg = Config(D=128, MCOLS=mcols, ACT_ROWS=8, DRAM_BYTES=32 << 20)
    engine = Engine(spec, weights(), cap=128, cfg=cfg, resident=True, pipeline=False, rows=1)
    engine.step(3)
    size = engine.image.nbytes
    isa = engine.backend
    rtl = RtlBackend(cfg, [sl.dram[:size] for sl in isa.machine.slices])
    expected = engine.step(17)
    engine.backend, engine.pos = rtl, engine.pos - 1
    actual = engine.step(17)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(isa.machine.slices[0].dram[:size], rtl.drams[0][:size])
