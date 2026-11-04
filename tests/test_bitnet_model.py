import numpy as np
import pytest

from opentpu.bitnet import pack_hf_bitnet
from opentpu.bitnet_model import (BitNetConfig, BitNetModel, IsaBackend, RefBackend,
                                  Safetensors, rmsnorm)
from opentpu.isasim import Config


class FakeStore(dict):
    def get(self, name):
        return self[name]


def tiny(layers=2, seed=3):
    rng = np.random.default_rng(seed)
    c = BitNetConfig(hidden=32, ffn=64, layers=layers, heads=4, kv_heads=2, vocab=50,
                     rope_theta=500000.0, eps=1e-5, head_dim=8)
    st = FakeStore()
    st["model.embed_tokens.weight"] = rng.standard_normal((c.vocab, c.hidden), dtype=np.float32)
    st["model.norm.weight"] = 1 + 0.1 * rng.standard_normal(c.hidden, dtype=np.float32)
    shapes = {"q_proj": (32, 32), "k_proj": (16, 32), "v_proj": (16, 32), "o_proj": (32, 32),
              "gate_proj": (64, 32), "up_proj": (64, 32), "down_proj": (32, 64)}
    for i in range(layers):
        p = f"model.layers.{i}."
        for nm, n in (("input_layernorm", 32), ("post_attention_layernorm", 32),
                      ("self_attn.attn_sub_norm", 32), ("mlp.ffn_sub_norm", 64)):
            st[p + nm + ".weight"] = 1 + 0.1 * rng.standard_normal(n, dtype=np.float32)
        for pj, (n, k) in shapes.items():
            sub = "self_attn." if pj in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp."
            t = rng.integers(-1, 2, size=(n, k)).astype(np.int8)
            st[p + sub + pj + ".weight"] = pack_hf_bitnet(t)
            st[p + sub + pj + ".weight_scale"] = np.array([1.0 + 0.5 * rng.random()], np.float32)
    return c, st


def test_rmsnorm():
    x = np.array([[3.0, 4.0]], np.float32)
    assert np.allclose(rmsnorm(x, np.ones(2, np.float32), 0.0), x / np.sqrt(12.5))


def test_tmm_rows_are_independent_of_each_others_offsets():
    # one row with a tiny element (QACT offset d > 11) must not change the other rows
    from opentpu import fp32 as F
    from opentpu.bitnet import pack_hf_bitnet
    rng = np.random.default_rng(5)
    cfg = Config(S=1, D=8, MCOLS=4, ACT_BLOCKS=16, TMEM_WORDS=1024, DRAM_BYTES=1 << 24, LANES=4)
    t = rng.integers(-1, 2, size=(32, 64)).astype(np.int8)
    be = IsaBackend(cfg)
    be.add("w", pack_hf_bitnet(t), 1.0)
    x = rng.standard_normal((4, 64), dtype=np.float32)
    x[0, 40], x[0, 41] = 0.0, 2e-4
    q, s, d = F.quantize_mx(x, 2, False)
    assert d[0].max() > 11
    rec = (q.astype(np.float32).reshape(4, -1, 2)
           * (s[:, None, None] / np.ldexp(1.0, d.astype(int))[:, :, None])).reshape(4, 64)
    got = be.linear("w", x)
    for r in range(1, 4):
        want = rec[r] @ t.T.astype(np.float32)
        assert np.linalg.norm(got[r] - want) / np.linalg.norm(want) < 1e-5


def test_isa_backend_tracks_reference_model():
    c, st = tiny()
    cfg = Config(S=1, D=8, MCOLS=4, ACT_BLOCKS=16, TMEM_WORDS=1024, DRAM_BYTES=1 << 24, LANES=4)
    ref = BitNetModel(c, st, RefBackend())
    isa = BitNetModel(c, st, IsaBackend(cfg))
    prompt = [1, 7, 3, 9]
    a, b = ref.forward(prompt), isa.forward(prompt)
    assert np.linalg.norm(a - b) / np.linalg.norm(a) < 0.1
    a, b = ref.forward([5]), isa.forward([5])      # decode step uses the KV cache
    assert np.linalg.norm(a - b) / np.linalg.norm(a) < 0.1
    assert isa.be.runs > 0


def test_kv_cache_matches_full_prefill():
    c, st = tiny()
    m = BitNetModel(c, st, RefBackend())
    full = m.forward([1, 2, 3, 4, 5])
    m.reset()
    m.forward([1, 2, 3])
    inc = m.forward([4, 5])
    assert np.allclose(full, inc, atol=1e-3)


def test_safetensors_reader(tmp_path):
    import json
    import struct
    w = np.arange(6, dtype=np.float32)
    bf = (np.array([1.5, -2.0], np.float32).view(np.uint32) >> 16).astype("<u2")
    hdr = {"a": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
           "b": {"dtype": "BF16", "shape": [2], "data_offsets": [24, 28]}}
    h = json.dumps(hdr).encode()
    (tmp_path / "t.safetensors").write_bytes(struct.pack("<Q", len(h)) + h + w.tobytes() + bf.tobytes())
    st = Safetensors(tmp_path / "t.safetensors")
    assert st.get("a").tolist() == [[0, 1, 2], [3, 4, 5]]
    assert st.get("b").tolist() == [1.5, -2.0]
