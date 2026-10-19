"""BitNet b1.58 on the Qwen3 engine (opentpu/llm/bitnet.py): every projection a TMM, the
sub-norms and relu^2 on the VPU, the int8 KV cache and LM head; the ISA simulator's logits
against HF transformers' BitNet (offline AutoBitLinear) on a tiny checkpoint in the real layout."""
import json
import struct

import numpy as np
import pytest

from opentpu.bitnet import pack_hf_bitnet
from opentpu.llm import load_spec
from opentpu.llm.bitnet import IMAGE, Weights
from opentpu.isasim import board_config
from opentpu.llm.qwen3 import Engine

H, F, L, NH, KV, V, HD = 256, 512, 2, 2, 1, 512, 128


def bf16(a):
    u = np.asarray(a, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    path = tmp_path_factory.mktemp("bitnet")
    rng = np.random.default_rng(11)
    t = {"model.embed_tokens.weight": ("BF16", bf16(rng.standard_normal((V, H)) * 0.5)),
         "model.norm.weight": ("BF16", bf16(1 + 0.1 * rng.standard_normal(H)))}
    shapes = {"self_attn.q_proj": (NH * HD, H), "self_attn.k_proj": (KV * HD, H),
              "self_attn.v_proj": (KV * HD, H), "self_attn.o_proj": (H, NH * HD),
              "mlp.gate_proj": (F, H), "mlp.up_proj": (F, H), "mlp.down_proj": (H, F)}
    for i in range(L):
        p = f"model.layers.{i}."
        for nm, n in (("input_layernorm", H), ("post_attention_layernorm", H),
                      ("self_attn.attn_sub_norm", NH * HD), ("mlp.ffn_sub_norm", F)):
            t[p + nm + ".weight"] = ("BF16", bf16(1 + 0.1 * rng.standard_normal(n)))
        for pj, (n, k) in shapes.items():
            w = rng.choice([-1, 0, 1], size=(n, k), p=[.25, .5, .25]).astype(np.int8)
            t[p + pj + ".weight"] = ("U8", pack_hf_bitnet(w))
            t[p + pj + ".weight_scale"] = ("BF16", bf16([0.5 + 2.0 * rng.random()]))
    hdr, blobs, off = {}, [], 0
    for k, (dt, a) in t.items():
        b = np.ascontiguousarray(a).tobytes()
        hdr[k] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))
    cfg = {"architectures": ["BitNetForCausalLM"], "model_type": "bitnet", "hidden_size": H,
           "intermediate_size": F, "num_hidden_layers": L, "num_attention_heads": NH,
           "num_key_value_heads": KV, "vocab_size": V, "head_dim": HD, "rope_theta": 500000.0,
           "rms_norm_eps": 1e-5, "hidden_act": "relu2", "max_position_embeddings": 512,
           "tie_word_embeddings": True, "bos_token_id": 1, "eos_token_id": 2,
           "quantization_config": {"quant_method": "bitnet", "linear_class": "autobitlinear",
                                   "quantization_mode": "offline"}}
    (path / "config.json").write_text(json.dumps(cfg))
    return path


def _cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


TOKS = [1, 5, 9, 33, 64, 7, 200, 311, 17, 3]


def test_spec(ckpt):
    spec = load_spec(ckpt)
    assert spec.subln and spec.act == "relu2" and not spec.qk_norm and spec.tied


def test_engine_matches_numpy_reference(ckpt):
    from opentpu.bitnet_model import BitNetConfig, BitNetModel, RefBackend, Safetensors
    spec, W = load_spec(ckpt), Weights(ckpt)
    eng = Engine(spec, W, cap=256, cfg=board_config(DRAM_BYTES=1 << 26), pipeline=False, **IMAGE)
    dev = np.array([eng.step(t) for t in TOKS])
    ref = BitNetModel(BitNetConfig.from_json(ckpt / "config.json"),
                      Safetensors(ckpt / "model.safetensors"), RefBackend())
    want = np.array([ref.forward([t]) for t in TOKS])
    c = _cos(dev, want)
    assert c.min() > 0.99, c
    assert (dev.argmax(1) == want.argmax(1)).mean() >= 0.8


def test_engine_matches_transformers(ckpt):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM
    spec, W = load_spec(ckpt), Weights(ckpt)
    m = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        hf = m(torch.tensor([TOKS])).logits[0].numpy()
    eng = Engine(spec, W, cap=256, cfg=board_config(DRAM_BYTES=1 << 26), pipeline=False, **IMAGE)
    dev = np.array([eng.step(t) for t in TOKS])
    c = _cos(dev, hf)
    assert c.min() > 0.99, c
