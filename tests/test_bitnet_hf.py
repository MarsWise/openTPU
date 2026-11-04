"""BitNet b1.58 against the framework: HF transformers' BitNet on a tiny synthetic checkpoint in the
real checkpoint's layout (U8 packed [N/4, K] weights, bf16 weight_scale and norms, the bitnet
quantization_config). Skipped without torch and transformers."""
import json
import struct

import numpy as np
import pytest

from opentpu.bitnet import pack_hf_bitnet
from opentpu.bitnet_model import BitNetConfig, BitNetModel, RefBackend, Safetensors

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

H, F, L, NH, KV, V, HD = 64, 128, 2, 4, 2, 97, 16


def bf16(a):
    u = np.asarray(a, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def checkpoint(path, linear_class, seed=7):
    rng = np.random.default_rng(seed)
    t = {"model.embed_tokens.weight": ("BF16", bf16(rng.standard_normal((V, H)) * 0.5)),
         "model.norm.weight": ("BF16", bf16(1 + 0.1 * rng.standard_normal(H)))}
    shapes = {"self_attn.q_proj": (NH * HD, H), "self_attn.k_proj": (KV * HD, H),
              "self_attn.v_proj": (KV * HD, H), "self_attn.o_proj": (H, NH * HD),
              "mlp.gate_proj": (F, H), "mlp.up_proj": (F, H), "mlp.down_proj": (H, F)}
    for i in range(L):
        p = f"model.layers.{i}."
        for nm, n in (("input_layernorm", H), ("post_attention_layernorm", H),
                      ("self_attn.attn_sub_norm", H), ("mlp.ffn_sub_norm", F)):
            t[p + nm + ".weight"] = ("BF16", bf16(1 + 0.1 * rng.standard_normal(n)))
        for pj, (n, k) in shapes.items():
            w = rng.choice([-1, 0, 1], size=(n, k), p=[.25, .5, .25]).astype(np.int8)
            t[p + pj + ".weight"] = ("U8", pack_hf_bitnet(w))
            t[p + pj + ".weight_scale"] = ("BF16", bf16([1.0 + 3.0 * rng.random()]))
    hdr, blobs, off = {}, [], 0
    for k, (dt, a) in t.items():
        b = np.ascontiguousarray(a).tobytes()
        hdr[k] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(path / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))
    cfg = {"architectures": ["BitNetForCausalLM"], "model_type": "bitnet", "hidden_size": H,
           "intermediate_size": F, "num_hidden_layers": L, "num_attention_heads": NH,
           "num_key_value_heads": KV, "vocab_size": V, "head_dim": HD, "rope_theta": 500000.0,
           "rms_norm_eps": 1e-5, "hidden_act": "relu2", "max_position_embeddings": 256,
           "tie_word_embeddings": True, "bos_token_id": 1, "eos_token_id": 2,
           "quantization_config": {"quant_method": "bitnet", "linear_class": linear_class,
                                   "quantization_mode": "offline"}}
    (path / "config.json").write_text(json.dumps(cfg))
    return path


def hf_logits(path, steps):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()
    out, past = [], None
    with torch.no_grad():
        for s in steps:
            o = m(torch.tensor([s]), past_key_values=past, use_cache=True)
            past = o.past_key_values
            out.append(o.logits[0, -1].numpy())
    return out


def ours(path, steps):
    model = BitNetModel(BitNetConfig.from_json(path / "config.json"),
                        Safetensors(path / "model.safetensors"), RefBackend())
    return [model.forward(s) for s in steps]


STEPS = [[1, 5, 9, 33, 64, 7], [11], [42]]           # a prefill, then two decode steps on the cache


def test_reference_matches_transformers_autobitlinear(tmp_path):
    p = checkpoint(tmp_path, "autobitlinear")
    for a, b in zip(hf_logits(p, STEPS), ours(p, STEPS)):
        assert np.abs(a - b).max() <= 1e-4 * np.abs(a).max()
        assert a.argmax() == b.argmax()


def test_bitlinear_is_a_different_convention(tmp_path):
    # BitLinear divides by weight_scale where AutoBitLinear multiplies: the reference is only
    # AutoBitLinear's, so a checkpoint must be checked for its linear_class before use
    p = checkpoint(tmp_path, "bitlinear")
    a, b = hf_logits(p, STEPS[:1])[0], ours(p, STEPS[:1])[0]
    assert np.abs(a - b).max() > 1e-2 * np.abs(a).max()
