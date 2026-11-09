"""Ouro (ByteDance Seed's looped language model, model_type "ouro": Ouro-1.4B / 2.6B) on the
device graph: the qwen3 Image's layer stack run `steps` times (total_ut_steps, 4), as
modeling_ouro.py does (read, not executed):

    h = embed(tokens)
    for ut in range(steps):
        for each layer:  h = h + norm2(attn(norm1(h)))          (input_layernorm, _2)
                         h = h + norm4(mlp(norm3(h)))           (post_attention_layernorm, _2)
        h = norm(h)                                             (model.norm, after every step)
    logits = lm_head(h)                                         (the last step's: early exit
                                                                 off -- threshold 1.0 reaches
                                                                 the last step)

The weights are stored once (one block per layer, as qwen3's); the KV cache is per (step,
layer) (UniversalTransformerCache's index ut * L + layer): step 0's in the layer blocks,
steps 1.. in a KV area after the image (per step, per layer the block's KV span). The second
norms of each layer (input_layernorm_2, post_attention_layernorm_2) are a table [L, 2, H]
after the image (qwen3._attention / _mlp's `post`: RMSNorm of the sublayer output before the
residual add). No q / k norms, plain RoPE (theta 1e6), untied head.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .. import fp32 as F
from .. import language as ol
from ..compiler import Affine, KVDesc, Tensor, trace_fit
from . import qwen3 as Q3
from .qwen3 import (ATTN_BLOCK, _attention, _inputs, _layers, _lm_head, _mlp, _tok_arg,
                    fill_logits, rmsnorm, rope_tables, step_descriptors)


@dataclass(frozen=True)
class Spec(Q3.Spec):
    steps: int = 4                  # total_ut_steps: the layer stack's passes

    @staticmethod
    def from_hf(model_dir, steps: int | None = None) -> "Spec":
        c = json.loads((Path(model_dir) / "config.json").read_text())
        if c.get("model_type") != "ouro" or c.get("hidden_act") != "silu":
            raise ValueError("expected an Ouro SiLU checkpoint")
        L = int(c["num_hidden_layers"])
        if c.get("layer_types", ["full_attention"] * L) != ["full_attention"] * L \
                or c.get("use_sliding_window") or c.get("rope_scaling"):
            raise ValueError("Ouro: full attention, plain RoPE only")
        return Spec(hidden=c["hidden_size"], layers=L, n_q=c["num_attention_heads"],
                    n_kv=c["num_key_value_heads"],
                    head_dim=c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
                    ffn=c["intermediate_size"], vocab=c["vocab_size"], eps=c["rms_norm_eps"],
                    theta=c.get("rope_theta", 1e6), tied=bool(c.get("tie_word_embeddings")),
                    bos=c.get("bos_token_id", 1), eos=(c.get("eos_token_id", 2),),
                    qk_norm=False, steps=int(c.get("total_ut_steps", 4) if steps is None
                                             else steps))

    def image(self, cfg, cap: int, batch: int = 1, rows: int = 1, wformat: str = "int8",
              head_format=None, lookup: bool = False, embed_host=None, formats=None,
              steering: bool = False, kv=None) -> "Image":
        return Image(self, cfg, cap, batch, rows, wformat, head_format, lookup, embed_host,
                     formats, steering, kv)


def reference_logits(spec: Spec, W, tokens) -> np.ndarray:
    """fp32 numpy forward of the whole sequence (causal), modeling_ouro.py's: logits [T, V]."""
    tokens = list(tokens)
    T, d, G = len(tokens), spec.head_dim, spec.n_q // spec.n_kv
    x = np.asarray(W["model.embed_tokens.weight"][tokens], np.float32)

    def norm(v, g):
        return (v / np.sqrt(np.mean(v * v, axis=-1, keepdims=True) + spec.eps)) * g

    cs = [rope_tables(spec, p) for p in range(T)]
    cos = np.stack([c for c, _ in cs])[:, None, :]
    sin = np.stack([s for _, s in cs])[:, None, :]

    def rot(v):
        h = d // 2
        return np.concatenate([v[..., :h] * cos - v[..., h:] * sin,
                               v[..., h:] * cos + v[..., :h] * sin], axis=-1)

    mask = np.triu(np.full((T, T), -np.inf, np.float32), 1)
    for _ in range(spec.steps):
        for i in range(spec.layers):
            p = f"model.layers.{i}."
            h = norm(x, W[p + "input_layernorm.weight"])
            q = rot((h @ W[p + "self_attn.q_proj.weight"].T).reshape(T, spec.n_q, d))
            k = rot((h @ W[p + "self_attn.k_proj.weight"].T).reshape(T, spec.n_kv, d))
            v = (h @ W[p + "self_attn.v_proj.weight"].T).reshape(T, spec.n_kv, d)
            o = np.zeros((T, spec.n_q, d), np.float32)
            for hq in range(spec.n_q):
                s = q[:, hq] @ k[:, hq // G].T / math.sqrt(d) + mask
                s = np.exp(s - s.max(axis=1, keepdims=True))
                o[:, hq] = (s / s.sum(axis=1, keepdims=True)) @ v[:, hq // G]
            a = o.reshape(T, -1) @ W[p + "self_attn.o_proj.weight"].T
            x = x + norm(a, W[p + "input_layernorm_2.weight"])
            h = norm(x, W[p + "post_attention_layernorm.weight"])
            g = h @ W[p + "mlp.gate_proj.weight"].T
            u = h @ W[p + "mlp.up_proj.weight"].T
            y = ((g / (1 + np.exp(-g))) * u) @ W[p + "mlp.down_proj.weight"].T
            x = x + norm(y, W[p + "post_attention_layernorm_2.weight"])
        x = norm(x, W["model.norm.weight"])
    head = W["model.embed_tokens.weight"] if spec.tied else W["lm_head.weight"]
    return x @ np.asarray(head, np.float32).T


class Image(Q3.Image):
    """qwen3.Image plus Ouro's second norms (a table [L, 2, H]) and the KV areas of steps
    1 .. steps-1 (per step L spans of the layer block's KV, each laid out as the block's)."""

    def __init__(self, spec: Spec, cfg, cap, batch=1, rows=1, wformat="int8", head_format=None,
                 lookup=False, embed_host=None, formats=None, steering=False, kv=None):
        if steering:
            raise ValueError("Ouro: no steering vectors")
        super().__init__(spec, cfg, cap, batch, rows, wformat, head_format, lookup, embed_host,
                         formats, steering, kv)
        if len(set(self.lf)) != 1:
            raise ValueError("Ouro: one weight format for every layer (one block layout)")
        H, L = spec.hidden, spec.layers
        offs = [v for heads in self.lofs["kvs"] for r in heads for v in r.values()]
        self.kv_lo = min(offs)
        self.kv_span = (self.LS - self.kv_lo + 4095) // 4096 * 4096   # (the block's tail)
        b = (self.nbytes + 4095) // 4096 * 4096
        self.norms2 = b
        b += L * 8 * H
        b = (b + 4095) // 4096 * 4096
        self.kv_areas = [None]
        for _ in range(1, spec.steps):
            self.kv_areas.append(b)
            b += L * self.kv_span
        self.nbytes = (b + 4095) // 4096 * 4096
        if self.nbytes > cfg.DRAM_BYTES:
            raise MemoryError(f"model image needs {self.nbytes / 2**20:.0f} MiB per slice, "
                              f"DRAM_BYTES is {cfg.DRAM_BYTES / 2**20:.0f} MiB")

    def build(self, W) -> list:
        imgs = super().build(W)
        H = self.spec.hidden
        for i in range(self.spec.layers):
            p = f"model.layers.{i}."
            t = np.concatenate([np.asarray(W[p + "input_layernorm_2.weight"], np.float32),
                                np.asarray(W[p + "post_attention_layernorm_2.weight"],
                                           np.float32)])
            v = F.ftz(t).view(np.uint8)
            for img in imgs:
                img[self.norms2 + i * 8 * H:self.norms2 + (i + 1) * 8 * H] = v
        return imgs

    def descriptors(self, sid: int) -> SimpleNamespace:
        m = super().descriptors(sid)
        spec, cfg, H, d = self.spec, self.cfg, self.spec.hidden, self.spec.head_dim
        ulen = {}
        for first, unit, reps in self.runs:
            for e in range(len(unit)):
                ulen[first + e] = len(unit)
        base_layer = m.layer

        def layer(li, it=None, ut: int = 0):
            lw = base_layer(li, it)
            idx = Affine.of(li) + (0 if it is None else Affine.of(it) * ulen[li])
            g = Affine(self.norms2) + idx * (8 * H)
            lw.g_in2 = Tensor(g, (H,), (1,))
            lw.g_post2 = Tensor(g + 4 * H, (H,), (1,))
            if ut:
                off = Affine(self.kv_areas[ut] - self.kv_lo) + idx * self.kv_span
                lw.kvs = [KVDesc({sid + j * cfg.S: {k: off + v for k, v in r.items()}
                                  for j, r in enumerate(heads)}, self.cap, self.dk, cfg.D,
                                 cfg.S, sid, dv=d,
                                 kfmt="int8" if self.kv == "int8" else "kv6",
                                 vfmt="kv6" if self.kv == "k6v6" else "int8",
                                 vones=self.io.get("v6ones"), vplanes=self.v6planes)
                          for heads in self.lofs["kvs"]]
                lw.kv = lw.kvs[0]
            return lw

        m.layer = layer
        m.steps = spec.steps
        return m

    def compile_step(self, pos: int, block: int = ATTN_BLOCK, tok: int | None = None) -> list:
        return [trace_fit(ouro_step, self.cfg, s,
                          lambda s=s: {"m": step_descriptors(self, s), "pos": pos,
                                       "block": block, **_tok_arg(self, tok)}).finish()
                for s in range(self.cfg.S)]

    def compile_rows(self, *a, **k):
        raise NotImplementedError("Ouro: one token a run (compile_step)")

    def compile_decode(self, blocks: int, lo: int, block: int = ATTN_BLOCK):
        return Q3.compile_decode(self, ouro_step, blocks, lo, block)

    def compile_generate(self, blocks: int, lo: int, block: int = ATTN_BLOCK,
                         chain: bool = True, samp=None, debug: bool = False,
                         part: int | None = None) -> list:
        from . import generate as G
        return G.compile_generate(self, ouro_step, blocks, lo, block, chain, samp, debug, part)


class _Step:
    """m with layer() bound to one step's KV (for qwen3._layers)."""

    def __init__(self, m, ut):
        self.m, self.ut = m, ut

    def __getattr__(self, k):
        return getattr(self.m, k)

    def layer(self, li, it=None):
        return self.m.layer(li, it, self.ut)


@ol.jit
def ouro_step(m, pos: int, block: int = ATTN_BLOCK, tok: int | None = None):
    """One decode token at position `pos`: its embedding -> logits, the layer stack `steps`
    times (each its own KV cache), the final norm after every step (the last one: the head's)."""
    fill_logits(m)
    spec = m.spec
    x, c, s_ = _inputs(m, pos, tok)
    for ut in range(m.steps):
        for lw in _layers(_Step(m, ut)):
            x.set(_attention(x, lw, c, s_, pos, spec, block, post=lw.g_in2))
            x.set(_mlp(x, lw, spec, post=lw.g_post2))
        if ut < m.steps - 1:
            x.set(rmsnorm(x, ol.load(m.g_final), spec.eps))
    _lm_head(x, m, spec)
