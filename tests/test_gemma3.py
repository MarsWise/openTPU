"""Gemma 3 (gemma3_text: FunctionGemma 270M) through gemma4.Spec.from_gemma3 / Gemma3Weights:
1 + w norm gains, the attention scale folded into q_norm, no V norm, no layer scalar; the device
program is gemma4's. A tiny random model saved by Hugging Face always runs."""
import json

import numpy as np
import pytest

from opentpu.isasim import board_config
from opentpu.llm import gemma4 as G
from opentpu.llm import load_spec
from opentpu.llm.qwen3 import Engine

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

S_, F_ = "sliding_attention", "full_attention"
KINDS = (S_, S_, F_, S_, F_)


def _cos(a, b):
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """FunctionGemma's shape at a small scale: 4 query heads on 1 KV head of 256, hidden 256,
    sliding window 512, local / global RoPE 1e4 / 1e6, query_pre_attn_scalar 256."""
    d = tmp_path_factory.mktemp("gemma3")
    torch.manual_seed(0)
    hc = transformers.Gemma3TextConfig(
        hidden_size=256, num_hidden_layers=len(KINDS), num_attention_heads=4,
        num_key_value_heads=1, head_dim=256, intermediate_size=512, vocab_size=1000,
        layer_types=list(KINDS), sliding_window=512, query_pre_attn_scalar=256,
        rope_theta=1e6, rope_local_base_freq=1e4, max_position_embeddings=4096,
        final_logit_softcapping=None, attn_logit_softcapping=None)
    hc._attn_implementation = "eager"
    m = transformers.Gemma3ForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:                       # (Gemma 3's gains are 1 + w)
                p.copy_(0.1 * torch.randn_like(p))
    m.save_pretrained(d)
    cfg = json.loads((d / "config.json").read_text())
    assert cfg["model_type"] == "gemma3_text"
    spec = load_spec(d)
    return m, G.load_weights(d), spec


def _cfg():
    return board_config(DRAM_BYTES=1 << 26)


def test_spec_from_gemma3(tiny):
    _, W, spec = tiny
    assert isinstance(W, G.Gemma3Weights) and not spec.v_norm and spec.ple_dim == 0
    assert spec.kinds == tuple(G.FULL if k == F_ else G.SLIDE for k in KINDS)
    assert spec.global_rot == 128 and spec.theta == 1e4 and spec.global_theta == 1e6
    assert np.all(W["model.layers.0.layer_scalar"] == 1)


def test_reference_matches_hf(tiny):
    m, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 40)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    assert np.abs(G.reference_logits(spec, W, toks) - hf).max() < 1e-4


def test_device_matches_hf_and_emulation(tiny):
    """The device program on the ISA simulator (int8 weights, fp4 for nothing here)."""
    m, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 20)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    eng = Engine(spec, W, cap=1024, cfg=_cfg())
    dev = np.array([eng.step(t) for t in toks])
    assert _cos(dev, hf).min() > 0.985
    emu = G.emulated_logits(spec, W, toks[:10])
    assert _cos(dev[:10], emu).min() > 0.99


@pytest.mark.slow
def test_token_on_board_rtl_is_bit_exact(have_verilator, tiny):
    """A resident token at position 520 (the sliding window's first block masked) on the
    Verilator RTL through the board's memory path equals the ISA simulator bit for bit: the
    logits and the whole DRAM image."""
    from opentpu import rtlsim
    from opentpu.llm.rtl_backend import RtlBackend
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(5).integers(0, 1000, 521)]
    eng = Engine(spec, W, cap=1024, cfg=_cfg(), wformat="int8", head_format="int8",
                 resident=True)
    eng.prefill(toks[:-1])
    n, isa = eng.image.nbytes, eng.backend
    rtl = RtlBackend(eng.cfg, [s.dram[:n] for s in isa.machine.slices],
                     uarch=rtlsim.BOARD_UARCH, axi=True, boot=True)
    want = eng.step(toks[-1])
    eng.backend, eng.pos = rtl, eng.pos - 1
    got = eng.step(toks[-1])
    assert np.array_equal(want.view(np.uint32), got.view(np.uint32))
    assert np.array_equal(isa.machine.slices[0].dram[:n], rtl.drams[0][:n])
