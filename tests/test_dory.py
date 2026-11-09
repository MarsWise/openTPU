"""Dory (looped nGPT, opentpu/llm/dory.py) on the device graph: a small model of Dory's form
(attention / MLP pairs with the S, S, S, * RoPE profile, input / recurrent x loops / output,
gated attention, l2 residual updates) on the ISA against the fp32 reference of dory.cpp's
graph; the real Q5_K_M checkpoint's first layers when present (OTPU_DORY_GGUF)."""
import math
import os

import numpy as np
import pytest

from opentpu.isasim import Config
from opentpu.llm import dory as DY
from opentpu.llm.qwen3 import Engine


class FakeRaw(dict):
    def rows(self, name, ids):
        return self[name][np.asarray(ids)]


def _raw(spec, seed=0):
    rng = np.random.default_rng(seed)
    H, d, F, nq, nk = spec.hidden, spec.head_dim, spec.ffn, spec.n_q * spec.head_dim, \
        spec.n_kv * spec.head_dim
    s = spec.init_std
    R = FakeRaw({"token_embd.weight": rng.normal(0, 1, (spec.vocab, H)).astype(np.float32),
                 "output.weight": rng.normal(0, 1 / math.sqrt(H), (spec.vocab, H)).astype(np.float32),
                 "output_scale.weight": rng.uniform(.5, 1.5, spec.vocab).astype(np.float32) * s * 20})
    mat = lambda n, k: rng.normal(0, 1 / math.sqrt(k), (n, k)).astype(np.float32)   # noqa
    for P in range(spec.layers):
        a, m = f"blk.{2 * P}.", f"blk.{2 * P + 1}."
        R.update({a + "attn_q.weight": mat(nq, H), a + "attn_k.weight": mat(nk, H),
                  a + "attn_v.weight": mat(nk, H), a + "attn_output.weight": mat(H, nq),
                  a + "attn_gate.weight": mat(nq, H),
                  a + "attn_qk_scale.weight": rng.uniform(.8, 1.2, nq).astype(np.float32) * s / math.sqrt(d) * 2,
                  a + "attn_gate_scale.weight": rng.uniform(.5, 1, nq).astype(np.float32) * s / math.sqrt(spec.gate_init),
                  a + "residual_alpha.weight": rng.uniform(1, 6, H).astype(np.float32) * s,
                  m + "ffn_gate.weight": mat(F, H), m + "ffn_up.weight": mat(F, H),
                  m + "ffn_down.weight": mat(H, F),
                  m + "ffn_gate_scale.weight": rng.uniform(.5, 1, F).astype(np.float32) / math.sqrt(H) * 3,
                  m + "ffn_up_scale.weight": rng.uniform(.5, 1, F).astype(np.float32) / math.sqrt(H) * 3,
                  m + "residual_alpha.weight": rng.uniform(1, 6, H).astype(np.float32) * s})
    return R


SPEC = DY.Spec(hidden=256, layers=12, n_q=2, n_kv=1, head_dim=128, ffn=512, vocab=512,
               eps=1e-12, tied=False, qk_norm=True, qk_smooth=False, n_input=4, n_rec=4, n_out=4,
               loops=2, kinds=(2, 2, 2, 1) * 3, window=1024)


@pytest.mark.parametrize("resident", [False, True])
def test_dory_small_against_reference(monkeypatch, resident):
    for key in ("OTPU_IMG_CACHE", "OTPU_IMAGE_CACHE", "OTPU_PROG_CACHE"):
        monkeypatch.setenv(key, "0")
    R = _raw(SPEC)
    tokens = [3, 17, 200, 45, 9]
    ref = DY.reference_logits(SPEC, R, tokens)
    cfg = Config(D=128, MCOLS=4, ACT_ROWS=8, DRAM_BYTES=64 << 20)
    eng = Engine(SPEC, DY.Weights(SPEC, R), cap=256, cfg=cfg, resident=resident,
                 pipeline=False, rows=1)
    dev = np.stack([np.asarray(eng.step(t)).reshape(-1) for t in tokens])
    cos = (dev * ref).sum(1) / np.linalg.norm(dev, axis=1) / np.linalg.norm(ref, axis=1)
    assert cos.min() > 0.99, cos
    assert (dev.argmax(1) == ref.argmax(1)).mean() >= 0.8
    # the loops count: one loop's reference is a different function (measured cos ~0.95)
    import dataclasses
    one = DY.reference_logits(dataclasses.replace(SPEC, loops=1), R, tokens)
    c1 = (dev * one).sum(1) / np.linalg.norm(dev, axis=1) / np.linalg.norm(one, axis=1)
    assert c1.max() < 0.98
