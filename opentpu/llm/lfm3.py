"""LFM3 (LiquidAI LFM3-1.7B, and the text decoder of LFM3-Omni): the model spec and its float
reference (FRAME). The device graph builds on lfm2 (gated short convolutions, the attention of
qwen3) -- not yet here.

The architecture, as modeling_lfm3.py of the checkpoint folder defines it (read, not executed):

  layers     `layer_types`: "conv" (LFM2's gated short convolution, here conv_L_cache = 7 taps),
             "sliding_attention" (a causal window of `sliding_window` tokens: key j counts for
             query p when p - j < window) and "full_attention"
  operator   x = x + post_operator_norm(op(operator_norm(x)))       (use_sandwich_norm)
  ffn        x = x + post_ffn_norm(w2(silu(w1 h) * w3 h)), h = ffn_norm(x)
  conv       B, C, xx = in_proj(h); y = C * causal_depthwise_conv_k(B * xx); out_proj(y)
  attention  q, k: per-head RMSNorm (q_norm / k_norm), then RoPE; softmax(q k^T / sqrt(d)) v;
             each head's output times sigmoid(gate_proj(h))[head] (one gate a head); o_proj
  RoPE       per layer type (rope_parameters[type]): theta, and a zero-tail partial rotation:
             the first int(factor * d) // 2 of the d / 2 frequencies keep the full head's
             progression theta^(-2i/d), the rest are 0 (those pairs are not rotated); the
             rotation is rotate_half's (pair i with i + d/2), so factor 0.5 at d = 128 rotates
             dimensions 0..31 and 64..95
  final      embedding_norm (an RMSNorm after the last layer), the head tied to the embedding
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CONV, SLIDE, FULL = "conv", "sliding_attention", "full_attention"


@dataclass(frozen=True)
class Spec:
    hidden: int
    kinds: tuple                # per layer: CONV, SLIDE or FULL
    n_q: int
    n_kv: int
    head_dim: int
    ffn: int
    vocab: int
    conv_k: int = 7
    window: int = 256
    eps: float = 1e-5
    rope: dict = field(default_factory=dict)   # kind -> (theta, partial factor)
    sandwich: bool = True
    tied: bool = True
    bos: int = 1
    eos: tuple = (7,)
    prefix: str = "model."      # the checkpoint's text decoder ("model.language_model." in Omni)

    @property
    def layers(self) -> int:
        return len(self.kinds)

    @staticmethod
    def from_hf(model_dir) -> "Spec":
        """From config.json: an LFM3 text config, or an LFM3-Omni / VL config's text_config
        (its decoder then under model.language_model.)."""
        c = json.loads((Path(model_dir) / "config.json").read_text())
        prefix = "model."
        if "text_config" in c:
            c, prefix = c["text_config"], "model.language_model."
        if c.get("model_type") != "lfm3":
            raise ValueError(f"not an LFM3 text model: {c.get('model_type')}")
        rp = c.get("rope_parameters") or {}
        rope = {k: (float(v.get("rope_theta", 1e6)), float(v.get("partial_rotary_factor", 1.0)))
                for k, v in rp.items()}
        for k, v in rp.items():
            if v.get("rope_type", "default") != "default":
                raise ValueError(f"RoPE type {v.get('rope_type')} ({k}) is not supported")
        eos = c.get("eos_token_id", 7)
        return Spec(hidden=c["hidden_size"], kinds=tuple(c["layer_types"]),
                    n_q=c["num_attention_heads"], n_kv=c["num_key_value_heads"],
                    head_dim=c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
                    ffn=c["intermediate_size"], vocab=c["vocab_size"],
                    conv_k=c.get("conv_L_cache", 7), window=c.get("sliding_window", 256),
                    eps=c.get("norm_eps", 1e-5), rope=rope,
                    sandwich=bool(c.get("use_sandwich_norm", True)),
                    tied=c.get("tie_word_embeddings", True), bos=c.get("bos_token_id", 1),
                    eos=tuple(eos) if isinstance(eos, list) else (eos,), prefix=prefix)

    def inv_freq(self, kind: str) -> np.ndarray:
        """The d/2 inverse frequencies of a layer kind (zero-tail partial rotation)."""
        theta, factor = self.rope[kind]
        d = self.head_dim
        n = int(factor * d) // 2
        f = 1.0 / (theta ** (np.arange(0, 2 * n, 2, dtype=np.float32) / d))
        return np.concatenate([f, np.zeros(d // 2 - n, np.float32)]).astype(np.float32)

    def rope_tables(self, kind: str, p) -> tuple:
        """cos, sin [..., d] at positions p (rotate_half's layout: both halves the same)."""
        f = np.asarray(p, np.float32)[..., None] * self.inv_freq(kind)
        e = np.concatenate([f, f], -1)
        return np.cos(e).astype(np.float32), np.sin(e).astype(np.float32)


def _norm(v, g, eps):
    v = np.asarray(v, np.float32)
    return (v / np.sqrt(np.mean(v * v, axis=-1, keepdims=True) + eps)) * g


def _rot(v, cos, sin):
    h = v.shape[-1] // 2
    rh = np.concatenate([-v[..., h:], v[..., :h]], -1)
    return v * cos + rh * sin


def reference_logits(spec: Spec, W, tokens) -> np.ndarray:
    """fp32 numpy forward of the whole sequence (causal); logits [T, vocab]. W maps the
    checkpoint's names (spec.prefix ...) to arrays (a lazy loader's mapping is fine: each
    tensor is read once a layer)."""
    tokens = list(tokens)
    P = spec.prefix
    T, d, H, K = len(tokens), spec.head_dim, spec.hidden, spec.conv_k
    G = spec.n_q // spec.n_kv
    f32 = lambda a: np.asarray(a, np.float32)                       # noqa: E731
    emb = W[P + "embed_tokens.weight"]
    x = f32(np.stack([np.asarray(emb[t], np.float32) for t in tokens]))
    pos = np.arange(T)
    causal = pos[None, :] <= pos[:, None]
    for i, kind in enumerate(spec.kinds):
        p = f"{P}layers.{i}."
        h = _norm(x, f32(W[p + "operator_norm.weight"]), spec.eps)
        if kind == CONV:
            B, C, xx = np.split(h @ f32(W[p + "conv.in_proj.weight"]).T, 3, axis=1)
            bx = np.concatenate([np.zeros((K - 1, H), np.float32), B * xx])
            w = f32(W[p + "conv.conv.weight"])[:, 0, :]               # [H, K]; w[:, K-1]: now
            conv = sum(bx[k:k + T] * w[:, k] for k in range(K))
            y = (C * conv) @ f32(W[p + "conv.out_proj.weight"]).T
        else:
            a = p + "self_attn."
            q = (h @ f32(W[a + "q_proj.weight"]).T).reshape(T, spec.n_q, d)
            k = (h @ f32(W[a + "k_proj.weight"]).T).reshape(T, spec.n_kv, d)
            v = (h @ f32(W[a + "v_proj.weight"]).T).reshape(T, spec.n_kv, d)
            cos, sin = spec.rope_tables(kind, pos)
            q = _rot(_norm(q, f32(W[a + "q_norm.weight"]), spec.eps), cos[:, None], sin[:, None])
            k = _rot(_norm(k, f32(W[a + "k_norm.weight"]), spec.eps), cos[:, None], sin[:, None])
            ok = causal & ((pos[:, None] - pos[None, :]) < spec.window) if kind == SLIDE \
                else causal
            mask = np.where(ok, 0.0, -np.inf).astype(np.float32)
            gate = 1.0 / (1.0 + np.exp(-(h @ f32(W[a + "gate_proj.weight"]).T)))   # [T, n_q]
            o = np.zeros((T, spec.n_q, d), np.float32)
            for hq in range(spec.n_q):
                s = q[:, hq] @ k[:, hq // G].T / math.sqrt(d) + mask
                s = np.exp(s - s.max(axis=1, keepdims=True))
                o[:, hq] = (s / s.sum(axis=1, keepdims=True)) @ v[:, hq // G]
            o = o * gate[:, :, None]
            y = o.reshape(T, -1) @ f32(W[a + "o_proj.weight"]).T
        if spec.sandwich:
            y = _norm(y, f32(W[p + "post_operator_norm.weight"]), spec.eps)
        x = x + y
        h = _norm(x, f32(W[p + "ffn_norm.weight"]), spec.eps)
        g = h @ f32(W[p + "feed_forward.w1.weight"]).T
        u = h @ f32(W[p + "feed_forward.w3.weight"]).T
        y = ((g / (1 + np.exp(-g))) * u) @ f32(W[p + "feed_forward.w2.weight"]).T
        if spec.sandwich:
            y = _norm(y, f32(W[p + "post_ffn_norm.weight"]), spec.eps)
        x = x + y
    x = _norm(x, f32(W[P + "embedding_norm.weight"]), spec.eps)
    head = W[P + "embed_tokens.weight"] if spec.tied else W["lm_head.weight"]
    return x @ f32(head).T
