"""Ouro's looped decoder on the device graph (opentpu/llm/ouro_dev.py): a small model of
Ouro's form (sandwich norms, untied head, no q / k norms) with its layer stack run 3 times,
each pass its own KV cache, the final norm after every pass -- the ISA against the fp32
reference (modeling_ouro.py's forward), at compile-time and run-time positions; and the
real Ouro-1.4B's first layers when its weights are present (OTPU_OURO_DIR)."""
import os

import numpy as np
import pytest

from opentpu.isasim import Config
from opentpu.llm import ouro_dev as O
from opentpu.llm.qwen3 import Engine


def _weights(spec, seed=0):
    rng = np.random.default_rng(seed)
    H, d, F = spec.hidden, spec.head_dim, spec.ffn
    W = {"model.embed_tokens.weight": rng.normal(0, 1, (spec.vocab, H)).astype(np.float32),
         "lm_head.weight": rng.normal(0, .05, (spec.vocab, H)).astype(np.float32),
         "model.norm.weight": rng.uniform(.8, 1.2, H).astype(np.float32)}
    for i in range(spec.layers):
        p = f"model.layers.{i}."
        for n in ("input_layernorm", "input_layernorm_2", "post_attention_layernorm",
                  "post_attention_layernorm_2"):
            W[p + n + ".weight"] = rng.uniform(.5, 1.5, H).astype(np.float32)
        for n, shape in (("self_attn.q_proj", (spec.n_q * d, H)),
                         ("self_attn.k_proj", (spec.n_kv * d, H)),
                         ("self_attn.v_proj", (spec.n_kv * d, H)),
                         ("self_attn.o_proj", (H, spec.n_q * d)),
                         ("mlp.gate_proj", (F, H)), ("mlp.up_proj", (F, H)),
                         ("mlp.down_proj", (H, F))):
            W[p + n + ".weight"] = rng.normal(0, 1 / np.sqrt(shape[1]), shape).astype(np.float32)
    return W


SPEC = O.Spec(hidden=256, layers=2, n_q=2, n_kv=1, head_dim=128, ffn=512, vocab=512,
              tied=False, bos=1, eos=(2,), qk_norm=False, steps=3)


@pytest.mark.parametrize("resident", [False, True])
def test_ouro_loop_against_reference(monkeypatch, resident):
    for key in ("OTPU_IMG_CACHE", "OTPU_IMAGE_CACHE", "OTPU_PROG_CACHE"):
        monkeypatch.setenv(key, "0")
    W = _weights(SPEC)
    tokens = [1, 7, 33, 100, 5]
    ref = O.reference_logits(SPEC, W, tokens)
    cfg = Config(D=128, MCOLS=4, ACT_ROWS=8, DRAM_BYTES=32 << 20)
    eng = Engine(SPEC, W, cap=256, cfg=cfg, resident=resident, pipeline=False, rows=1)
    dev = np.stack([np.asarray(eng.step(t)).reshape(-1) for t in tokens])
    cos = (dev * ref).sum(1) / np.linalg.norm(dev, axis=1) / np.linalg.norm(ref, axis=1)
    assert cos.min() > 0.99, cos
    assert (dev.argmax(1) == ref.argmax(1)).mean() >= 0.8
    # each pass its own cache: a model of one pass is a different function
    one = O.reference_logits(O.Spec(**{**SPEC.__dict__, "steps": 1}), W, tokens)
    assert np.abs(one - ref).max() > 1e-2


def test_ouro_refuses_mixed_formats():
    W = _weights(SPEC)
    cfg = Config(D=128, MCOLS=4, ACT_ROWS=8, DRAM_BYTES=32 << 20)
    with pytest.raises(ValueError):
        O.Image(SPEC, cfg, 256, formats="down@1-1=int4")


DIR = os.environ.get("OTPU_OURO_DIR", "/home/claude/ouro14")     # Ouro-1.4B-Thinking
IDS = [504, 3575, 282, 4649, 314]                                # "The capital of France is"


def _real():
    if not os.path.exists(os.path.join(DIR, "model.safetensors")):
        pytest.skip(f"no Ouro checkpoint in {DIR}")
    from opentpu.llm.qwen3 import load_weights
    return O.Spec.from_hf(DIR), load_weights(DIR)


def test_ouro_1_4b_real_first_layers_int8():
    """Real weights, the first 8 layers x 4 passes, int8: measured cos 0.99877 .. 0.99993."""
    import dataclasses
    from opentpu.isasim import board_config
    spec, W = _real()
    spec = dataclasses.replace(spec, layers=8)
    ref = O.reference_logits(spec, W, IDS)
    eng = Engine(spec, W, cap=256, cfg=board_config(DRAM_BYTES=1 << 31), resident=True,
                 pipeline=False, rows=1)
    dev = np.stack([np.asarray(eng.step(t)).reshape(-1) for t in IDS])
    cos = (dev * ref).sum(1) / np.linalg.norm(dev, axis=1) / np.linalg.norm(ref, axis=1)
    assert cos.min() > 0.998, cos
    assert list(np.argsort(-dev[-1])[:3]) == list(np.argsort(-ref[-1])[:3])


def test_ouro_1_4b_real_full_mlp4(monkeypatch):
    """All 24 layers x 4 passes, attention int8 and MLP int4 (the container's 8 GB): the next
    token is the reference's (" Paris"); measured cos 0.95 .. 0.99."""
    from opentpu.isasim import board_config
    monkeypatch.setenv("OTPU_FORMATS", "gateup=int4,down=int4")
    spec, W = _real()
    ref = O.reference_logits(spec, W, IDS)
    eng = Engine(spec, W, cap=256, cfg=board_config(DRAM_BYTES=1 << 31), resident=True,
                 pipeline=False, rows=1)
    dev = np.stack([np.asarray(eng.step(t)).reshape(-1) for t in IDS])
    cos = (dev * ref).sum(1) / np.linalg.norm(dev, axis=1) / np.linalg.norm(ref, axis=1)
    assert cos.min() > 0.94, cos
    assert dev[-1].argmax() == ref[-1].argmax() == 7042
