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


# =============================================================================== device graph
# LFM3 on openTPU builds on lfm2 (its Image, conv layers, decode step and Engine path):
#   * conv layers: lfm2._conv (conv_k = 7 taps, the mirrored state ring), and the sandwich
#     norm (post_operator_norm) on its output
#   * attention: qwen3._attention (q / k norms, GQA, int8 KV) with the head gate -- gate_proj's
#     row of each head repeated over the head's d dimensions, qwen3's `gated` (sigmoid of the
#     projection times the head's output) -- and the sandwich norm; RoPE from a table of the
#     layer's kind (full layers: theta 1e6, the zero-tail half rotation; sliding: theta 1e4,
#     whole), in DRAM per position; sliding layers attend over the last `window` tokens of the
#     cache (attention.Blocks, the first block masked at the window's start)
#   * MLP: qwen3._mlp with post_ffn_norm; the final norm is embedding_norm, the head tied
# The KV cache of a sliding layer keeps every position (cap slots), not a window ring.
# One slice (S = 1); decode steps at compile-time positions (compile_step), one token a run.
from types import SimpleNamespace  # noqa: E402

from .. import language as ol  # noqa: E402
from .. import quant as Q  # noqa: E402
from .. import qcache as QC  # noqa: E402
from .. import fp32 as F  # noqa: E402
from ..compiler import Affine, Tensor, trace_fit  # noqa: E402
from ..kernels.attention import Blocks  # noqa: E402
from . import lfm2 as L2  # noqa: E402
from .qwen3 import (ATTN_BLOCK, _attention, _inputs, _lm_head, _mlp, _tok_arg,  # noqa: E402
                    fill_logits, step_descriptors)


@dataclass(frozen=True)
class DevSpec(L2.Spec):
    """lfm2.Spec with LFM3's layer kinds (akinds: CONV, SLIDE or FULL), window, RoPE per kind
    (rope3: ((kind, theta, factor), ...)) and the checkpoint prefix."""
    akinds: tuple = ()
    window: int = 256
    rope3: tuple = ()
    prefix: str = "model."

    def check(self, cfg) -> None:
        super().check(cfg)
        if cfg.S != 1:
            raise ValueError("LFM3 runs on one slice (S = 1)")

    def image(self, cfg, cap: int, batch: int = 1, rows: int = 1, wformat: str = "int8",
              head_format=None, lookup=False, experts=None, embed_host=None,
              formats=None) -> "Image":
        if rows != 1:
            raise NotImplementedError("LFM3: one token a run (Engine(rows=1))")
        return Image(self, cfg, cap, batch, rows, wformat, head_format, lookup, experts,
                     embed_host, formats)

    def lfm3(self) -> Spec:
        return Spec(hidden=self.hidden, kinds=self.akinds, n_q=self.n_q, n_kv=self.n_kv,
                    head_dim=self.head_dim, ffn=self.ffn, vocab=self.vocab, conv_k=self.conv_k,
                    window=self.window, eps=self.eps,
                    rope={k: (t, f) for k, t, f in self.rope3}, tied=self.tied, bos=self.bos,
                    eos=self.eos, prefix=self.prefix)


def device_spec(spec: Spec) -> DevSpec:
    """The device's spec of an LFM3 model (lfm2's, with LFM3's kinds)."""
    if not spec.sandwich:
        raise ValueError("LFM3 without sandwich norms is not supported")
    return DevSpec(hidden=spec.hidden,
                   kinds=tuple(L2.CONV if k == CONV else L2.ATTN for k in spec.kinds),
                   n_q=spec.n_q, n_kv=spec.n_kv, head_dim=spec.head_dim, ffn=spec.ffn,
                   vocab=spec.vocab, conv_k=spec.conv_k, eps=spec.eps,
                   theta=spec.rope.get(FULL, (1e6, 1.0))[0], tied=spec.tied, bos=spec.bos,
                   eos=spec.eos, akinds=spec.kinds, window=spec.window,
                   rope3=tuple((k, t, f) for k, (t, f) in spec.rope.items()),
                   prefix=spec.prefix)


class DeviceWeights:
    """The checkpoint's LFM3 names as lfm2's (model.*, q_layernorm, k_layernorm, out_proj),
    lazily: the device image reads them through this mapping."""

    REN = (("self_attn.q_norm.", "self_attn.q_layernorm."),
           ("self_attn.k_norm.", "self_attn.k_layernorm."),
           ("self_attn.o_proj.", "self_attn.out_proj."))

    def __init__(self, W, prefix: str):
        self.W, self.prefix = W, prefix

    def _src(self, k: str) -> str:
        if k.startswith("model."):
            k = self.prefix + k[len("model."):]
        for a, b in self.REN:
            k = k.replace(b, a)
        return k

    def __getitem__(self, k):
        return self.W[self._src(k)]

    def __contains__(self, k):
        return self._src(k) in self.W

    def get(self, k, default=None):
        return self[k] if k in self else default


class Image(L2.Image):
    """lfm2.Image plus LFM3's parts: each layer's post_operator_norm and post_ffn_norm (after
    the lfm2 block's norms), the attention layers' gate rows (wgate [n_q d, H], gate_proj's
    head row repeated d times), the RoPE tables of both kinds [cap, d/2] and the sliding
    window's start mask (z2: block - 1 times -inf, then block times +inf). A layer's formats
    group carries its kind (CONV / SLIDE / FULL): hardware loops run layers of one kind."""

    def _groups(self, lf):
        return tuple(g + (k,) for g, k in zip(lf, self._akinds))

    def _layout(self, g, mlp, rb):
        if "wgate" not in self.mats[L2.ATTN]:
            self.mats[L2.ATTN]["wgate"] = (self.nq_loc * self.spec.head_dim, self.spec.hidden)
        ns = super()._layout(g[:4], mlp, rb)
        H = self.spec.hidden
        for kind in (L2.CONV, L2.ATTN):           # the sandwich norms at the block's end
            end = ns.size[kind]
            ns.lofs[kind]["g_pop"], ns.lofs[kind]["g_pffn"] = end, end + 4 * H
            ns.size[kind] = (end + 8 * H + 4095) // 4096 * 4096
        return ns

    def __init__(self, spec: DevSpec, cfg, cap, batch=1, rows=1, wformat="int8",
                 head_format=None, lookup=False, experts=None, embed_host=None, formats=None):
        # the gate rows are allocated in the attention block (_layout): lfm2's layout of wq's
        # format, and the plan's keys carry the layer kinds (_groups)
        self._akinds = spec.akinds
        super().__init__(spec, cfg, cap, batch, rows, wformat, head_format, lookup, experts,
                         embed_host, formats)
        d, blk = spec.head_dim, ATTN_BLOCK
        b = self.nbytes
        self.ropes = {}
        for k in (FULL, SLIDE):
            self.ropes[k] = (b, b + 4 * cap * (d // 2))
            b += 8 * cap * (d // 2)
        self.z2 = b
        b += 4 * (2 * blk - 1)
        self.nbytes = (b + 4095) // 4096 * 4096
        if self.nbytes > cfg.DRAM_BYTES:
            raise MemoryError(f"model image needs {self.nbytes / 2**20:.0f} MiB, DRAM_BYTES is "
                              f"{cfg.DRAM_BYTES / 2**20:.0f} MiB")

    def build(self, W) -> list:
        spec, cfg = self.spec, self.cfg
        D, d, H = cfg.D, spec.head_dim, spec.hidden
        dw = W if isinstance(W, DeviceWeights) else DeviceWeights(W, spec.prefix)
        imgs = super().build(dw)
        img = imgs[0]
        W, P = dw, "model."

        def put(addr, a):
            v = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
            img[addr:addr + v.size] = v

        def f32(a):
            return F.ftz(np.asarray(a, np.float32))

        qb = QC.Batch()
        for i, kind in enumerate(spec.akinds):
            p, base = f"{P}layers.{i}.", self._off(i).const
            g = self.layouts[self.lf[i]]
            lo = g.lofs[L2.CONV if kind == CONV else L2.ATTN]
            put(base + lo["g_pop"], f32(W[p + "post_operator_norm.weight"]))
            put(base + lo["g_pffn"], f32(W[p + "post_ffn_norm.weight"]))
            if kind != CONV:
                gate = np.repeat(np.asarray(W[p + "self_attn.gate_proj.weight"], np.float32),
                                 d, axis=0)                              # [n_q d, H]
                a0, a1 = lo["wgate"]
                qb.submit(gate, g.mf["wgate"], D,
                          lambda q, sc, a0=base + a0, a1=base + a1: (put(a0, q), put(a1, sc)))
        qb.wait()
        s3 = spec.lfm3()
        for k, (ca, sa) in self.ropes.items():
            c, s = s3.rope_tables(k, np.arange(self.cap))
            put(ca, f32(c[:, :d // 2]))
            put(sa, f32(s[:, :d // 2]))
        blk = ATTN_BLOCK
        put(self.z2, np.concatenate([np.full(blk - 1, -np.inf), np.full(blk, np.inf)])
            .astype(np.float32))
        return imgs

    def descriptors(self, sid: int) -> SimpleNamespace:
        m = super().descriptors(sid)
        spec, d, H, cap = self.spec, self.spec.head_dim, self.spec.hidden, self.cap
        base_layer = m.layer

        def layer(li, it=None, at=None):
            lw = base_layer(li, it, at)
            g = self.layouts[self.lf[li]]
            off = self._off(li, it)
            lo = g.lofs[lw.kind]
            lw.akind = spec.akinds[li]
            lw.g_pop = Tensor(off + lo["g_pop"], (H,), (1,))
            lw.g_pffn = Tensor(off + lo["g_pffn"], (H,), (1,))
            return lw

        m.layer = layer
        m.ropes = {k: (Tensor(Affine(c), (cap, d // 2), (d // 2, 1)),
                       Tensor(Affine(s), (cap, d // 2), (d // 2, 1)))
                   for k, (c, s) in self.ropes.items()}
        m.z2 = self.z2
        m.window = spec.window
        return m

    def compile_step(self, pos: int, block: int = ATTN_BLOCK, tok: int | None = None) -> list:
        return [trace_fit(lfm3_step, self.cfg, s,
                          lambda s=s: {"m": step_descriptors(self, s), "pos": pos,
                                       "block": block, **_tok_arg(self, tok)}).finish()
                for s in range(self.cfg.S)]

    def compile_rows(self, *a, **k):
        raise NotImplementedError("LFM3: one token a run (compile_step)")

    def compile_decode(self, *a, **k):
        raise NotImplementedError("LFM3: no resident decode yet (compile-time positions)")

    compile_generate = compile_prompt_run = compile_decode


def _slide_seq(m, p: int, block: int):
    """The cache tokens a sliding layer's query at position p attends to: the last m.window
    (p - window + 1 .. p) -- a length while that reaches position 0, else attention.Blocks over
    the window's blocks, the first masked before the window's start (a row of m.z2), the last
    ending at p."""
    W = m.window
    if p < W:
        return p + 1
    s = p - W + 1
    s0, k = s // block * block, s % block
    eb, t = p // block * block, p % block
    items = []
    for b0 in range(s0, eb + 1, block):
        n = block if b0 != eb else t + 1
        mask = Tensor(Affine(m.z2 + 4 * (block - 1 - k)), (n,), (1,)) if b0 == s0 and k else None
        items.append((b0, n, mask))
    return Blocks(items)


@ol.jit
def lfm3_step(m, pos: int, block: int = ATTN_BLOCK, tok: int | None = None):
    """One decode token at compile-time position `pos`: its embedding (the host's x, or the
    image's table at `tok`) -> logits. lfm2_step with LFM3's sandwich norms, head gates, RoPE
    tables per layer kind and sliding windows."""
    if not isinstance(pos, int):
        raise NotImplementedError("LFM3: compile-time positions only")
    fill_logits(m)
    spec = m.spec
    x, _, _ = _inputs(m, pos, tok)
    tabs = {k: (ol.load(c[pos, :]), ol.load(s[pos, :])) for k, (c, s) in m.ropes.items()}

    def layer(li, it):
        lw = m.layer(li, it)
        if lw.kind == L2.CONV:
            x.set(L2._conv(x, lw, pos, spec, post=lw.g_pop))
        else:
            c, s_ = tabs[lw.akind]
            seq = _slide_seq(m, pos, block) if lw.akind == SLIDE else None
            x.set(_attention(x, lw, c, s_, pos, spec, block, gated=True, seq=seq,
                             post=lw.g_pop))
        x.set(_mlp(x, lw, spec, post=lw.g_pffn))

    L2.run_layers(m.plan, layer)
    _lm_head(x, m, spec)
