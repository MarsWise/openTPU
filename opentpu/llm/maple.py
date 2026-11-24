"""Maple-Preview (deepgrove/maple-preview, its GGUF maple-preview-TQ2_0-head-F16.gguf): a 20B-A1B
ternary mixture-of-experts LLM, on openTPU as lfm2's attention + MoE layers (moe.py's path (a):
the card routes and computes every expert, the host streams the missing ones into DRAM slots).

The architecture, as the checkpoint's modeling_maple.py defines it (read as text, not executed)
and its GGUF's header states:

  layers     24, every one attention + a routed MoE (no dense MLP, no shared expert)
  attention  16 q heads, 4 kv heads, d 128, no biases; q_norm / k_norm (RMSNorm a head, gain
             w) before RoPE; softmax(q k^T / sqrt(d)) v; o_proj
  kinds      sliding_window_pattern: layers 4, 8, .. 24 (1-based) full attention without RoPE
             (nope_on_global_attention), the others a causal window of 512 tokens with RoPE
  RoPE       theta 1e4 on the first 64 dimensions of a head (rotate_half within them: pair
             (i, i + 32), frequency theta^(-2i/64)), the other 64 not rotated
  MoE        router F32 [256, H]: softmax over all 256, the 8 largest, renormalized (= the
             softmax of the 8 largest logits: moe.MoESpec rule "softmax"); experts SwiGLU of
             width 512 with silu(min(g, 7)) * clip(u, -7, 7) (swiglu_clamp_exp)
  norms      RMSNorm eps 1e-6, gain w (not zero-centered); attn_norm, ffn_norm, output_norm
  head       output.weight F16, untied from token_embd.weight F16; vocabulary 151936 (Qwen's)
  weights    attention and experts TQ2_0: blocks of 256 along a row, 2 bits a weight (q - 1
             in {-1, 0, +1}) and one fp16 scale d a block. Every 128-block of a row is then
             d * {-1, 0, +1}: quant.pack_ternary_gsc repacks them for TMM GSC losslessly.

On the device (lfm2.Image with Maple's parts, `Image`):

- the RoPE tables of both kinds (lfm3's form: a head's d / 2 frequencies, pair (i, i + d/2)).
  Maple's pairs (i, i + 32) over the first 64 dimensions become those by permuting each q and
  k head's dimensions (and the q / k norm gains with them; q . k does not change): new order
  0..31, 64..95, 32..63, 96..127 (`PERM`). Pair i < 32 then rotates at theta'^(-2i/128) with
  theta' = theta^2, the other 32 pairs at frequency 0 (cos 1, sin 0). The full layers' table
  is all frequency 0: no rotation (NoPE).
- the window: a context of at most 512 tokens (cap <= window) has every sliding layer attend
  to every earlier token, as a full layer does: exact. Longer contexts are refused here (the
  sliding window over several attention blocks at run-time positions is not built yet).

The GGUF's q / k rows are taken as Hugging Face's (rotate_half, not llama.cpp's interleaved
permutation: Maple's converter is qwen-like, NEOX RoPE). A wrong guess would show as the
reference's text going wrong (tools: r8_maple.sh's reference run), not silently.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .. import gguf as GG
from .. import language as ol
from .. import quant as Q
from .. import fp32 as F
from ..compiler import Affine, Tensor, trace_fit
from . import lfm2 as L2
from . import moe as MO
from .qwen3 import (ATTN_BLOCK, RunPos, _attention, _inputs, _lm_head, _tok_arg, fill_logits,
                    step_descriptors)
from . import generate as G

SLIDE, FULL = "sliding_attention", "full_attention"
D_HEAD = 128
PERM = np.r_[0:32, 64:96, 32:64, 96:128]        # new dimension n holds Maple's dimension PERM[n]


# ============================================================================ the GGUF's tensors
def tq2_values(raw: np.ndarray, rows: int, K: int) -> np.ndarray:
    """TQ2_0 bytes of `rows` rows of K -> fp32 [rows, K] (ggml's dequantize_row_tq2_0: a block
    is qs[64] then fp16 d; element 32 * (4 j + l) + m of the block is bits 2l of qs[32 j + m],
    value (q - 1) d)."""
    b = np.asarray(raw, np.uint8).reshape(rows * (K // 256), 66)
    d = b[:, 64:66].copy().view(np.float16).astype(np.float32)[:, 0]
    qs = b[:, :64].reshape(-1, 2, 32)                                   # [blk, j, m]
    q = np.stack([(qs >> (2 * l)) & 3 for l in range(4)], axis=2)       # [blk, j, l, m]
    if np.any(q == 3):
        raise ValueError("TQ2_0: reserved code 3")
    v = (q.reshape(-1, 256).astype(np.float32) - 1.0) * d[:, None]
    return v.reshape(rows, K)


def tq2_ternary(raw: np.ndarray, rows: int, K: int):
    """The same rows as int8 trits [rows, K] and fp32 scales [rows, K / 256]."""
    v = tq2_values(raw, rows, K)
    d = np.abs(v.reshape(rows, -1, 256)).max(axis=2)
    return np.sign(v).astype(np.int8), d


class GGUFWeights(Mapping):
    """The GGUF's tensors under their own names (blk.i.attn_q.weight, ...) as fp32 matrices
    [rows, K] (numpy order), read lazily; `expert(name, e)` reads one expert of a stacked
    [E, rows, K] tensor."""

    def __init__(self, path):
        self.g = GG.GGUF(path)
        self.kv = self.g.kv

    def __getitem__(self, n):
        ti = self.g.tensors[n]
        shape = tuple(reversed(ti.shape))
        if ti.type == GG.TQ2_0:
            if len(shape) != 2:
                raise ValueError(f"{n}: stacked TQ2_0, read it by expert()")
            return tq2_values(self.g.raw(n), shape[0], shape[1])
        return GG.dequant(self.g.raw(n), ti.type).reshape(shape)

    def rows(self, n: str, idx) -> np.ndarray:
        """Rows idx of an F16 / F32 matrix (the embedding's), without the whole table."""
        ti = self.g.tensors[n]
        K = ti.shape[0]
        if ti.type not in (GG.F16, GG.F32):
            return np.asarray(self[n][np.asarray(idx)], np.float32)
        b = 2 if ti.type == GG.F16 else 4
        raw = self.g.raw(n)
        return np.stack([GG.dequant(raw[i * K * b:(i + 1) * K * b], ti.type)
                         for i in np.asarray(idx).reshape(-1)])

    def expert(self, n: str, e: int) -> np.ndarray:
        ti = self.g.tensors[n]
        E, rows, K = tuple(reversed(ti.shape))
        per = rows * K // 256 * 66 if ti.type == GG.TQ2_0 else None
        raw = self.g.raw(n)
        if ti.type != GG.TQ2_0:
            return GG.dequant(raw, ti.type).reshape(E, rows, K)[e]
        return tq2_values(raw[e * per:(e + 1) * per], rows, K)

    def __iter__(self):
        return iter(self.g.tensors)

    def __len__(self):
        return len(self.g.tensors)


class DictWeights(dict):
    """GGUF-named weights in a dict (a tiny random model's): experts stacked [E, rows, K]."""

    def rows(self, n: str, idx) -> np.ndarray:
        return np.asarray(self[n][np.asarray(idx)], np.float32)

    def expert(self, n: str, e: int) -> np.ndarray:
        return np.asarray(self[n][e], np.float32)


# ================================================================================ the spec
@dataclass(frozen=True)
class Spec:
    hidden: int = 2048
    layers: int = 24
    n_q: int = 16
    n_kv: int = 4
    head_dim: int = D_HEAD
    rot: int = 64                   # RoPE dimensions of a head (the first ones)
    theta: float = 1e4
    window: int = 512
    full: tuple = (3, 7, 11, 15, 19, 23)     # 0-based layers of full attention (NoPE)
    E: int = 256
    k: int = 8
    ffn: int = 512                  # an expert's width
    clamp: float = 7.0
    vocab: int = 151936
    eps: float = 1e-6
    bos: int = 151643
    eos: tuple = (151645,)

    def kind(self, i: int) -> str:
        return FULL if i in self.full else SLIDE

    @staticmethod
    def from_gguf(kv: dict) -> "Spec":
        a = kv["general.architecture"]
        if a != "maple":
            raise ValueError(f"GGUF architecture {a!r}, not maple")
        g = lambda k: kv[f"maple.{k}"]                          # noqa: E731
        pat = g("attention.sliding_window_pattern")
        clamp = set(g("swiglu_clamp_exp"))
        if len(clamp) != 1:
            raise ValueError(f"per-layer SwiGLU clamps {sorted(clamp)}: one is supported")
        if not g("expert_weights_norm") or g("expert_weights_scale") != 1.0 \
                or g("expert_gating_func") != 1:
            raise ValueError("Maple's router: softmax, renormalized top-k, scale 1 expected")
        d = g("attention.key_length")
        if d != D_HEAD or g("attention.value_length") != d:
            raise ValueError(f"head dimension {d}: {D_HEAD} expected")
        if g("rope.dimension_count") != 64:
            raise ValueError("RoPE on 64 dimensions of 128 expected (PERM)")
        return Spec(hidden=g("embedding_length"), layers=g("block_count"),
                    n_q=g("attention.head_count"), n_kv=g("attention.head_count_kv"),
                    head_dim=d, rot=g("rope.dimension_count"), theta=float(g("rope.freq_base")),
                    window=g("attention.sliding_window"),
                    full=tuple(i for i, s in enumerate(pat) if not s),
                    E=g("expert_count"), k=g("expert_used_count"),
                    ffn=g("expert_feed_forward_length"), clamp=float(clamp.pop()),
                    vocab=g("vocab_size"), eps=float(g("attention.layer_norm_rms_epsilon")),
                    bos=kv.get("tokenizer.ggml.bos_token_id", 151643),
                    eos=(kv.get("tokenizer.ggml.eos_token_id", 151645),))

    # ---- the RoPE of Maple's own dimensions (the reference's)
    def rope(self, pos):
        inv = 1.0 / self.theta ** (np.arange(self.rot // 2, dtype=np.float64) * 2 / self.rot)
        f = np.multiply.outer(np.asarray(pos, np.float64), inv)
        return np.cos(f), np.sin(f)

    # ---- the device's form: lfm2.Spec and the permuted RoPE tables
    def moe(self) -> MO.MoESpec:
        if self.clamp != 7.0:
            raise ValueError(f"SwiGLU clamp {self.clamp}: moe.ACTS has silu_clamp7 only")
        return MO.MoESpec(E=self.E, k=self.k, ffn=self.ffn, first=0, rule="softmax", norm=True,
                          act="silu_clamp7")

    def device(self) -> "DevSpec":
        return DevSpec(hidden=self.hidden, kinds=(L2.ATTN,) * self.layers, n_q=self.n_q,
                       n_kv=self.n_kv, head_dim=self.head_dim, ffn=self.ffn, vocab=self.vocab,
                       eps=self.eps, theta=self.theta, tied=False, bos=self.bos, eos=self.eos,
                       moe=self.moe(), embed="int8", maple=self)

    def dev_tables(self, kind: str, pos) -> tuple:
        """cos, sin [len(pos), d/2] of the device's permuted dimensions (lfm3's form: pair i
        with i + d/2): pairs i < rot/2 at theta^2^(-2i/d) (= theta^(-2i/rot)), the rest 0; a
        full layer's all 0 (NoPE)."""
        h = self.head_dim // 2
        inv = np.zeros(h)
        if kind == SLIDE:
            r = self.rot // 2
            inv[:r] = 1.0 / self.theta ** (np.arange(r, dtype=np.float64) * 2 / self.rot)
        f = np.multiply.outer(np.asarray(pos, np.float64), inv)
        return np.cos(f), np.sin(f)


# ============================================================================= the reference
def _rms(v, g, eps):
    v = np.asarray(v, np.float64)
    return v / np.sqrt((v * v).mean(-1, keepdims=True) + eps) * g


def route(logits: np.ndarray, k: int):
    """modeling_maple.MapleGate: softmax over all, the k largest (ties to the lower index),
    renormalized. -> (ids, weights)."""
    p = np.exp(logits - logits.max())
    p /= p.sum()
    ids = np.argsort(-p, kind="stable")[:k]
    w = p[ids]
    return ids, w / (w.sum() + 1e-20)


def reference_logits(spec: Spec, W, tokens, routes: list | None = None) -> np.ndarray:
    """fp64 logits [T, vocab] of the GGUF-named weights W (GGUFWeights or DictWeights), Maple's
    own dimensions and RoPE, the sliding window, the clamped experts (only the chosen ones
    read). routes: appended (t, layer, ids, margin of the k-th over the (k+1)-th probability)."""
    H, d, nq, nkv = spec.hidden, spec.head_dim, spec.n_q, spec.n_kv
    T = len(tokens)
    rows = getattr(W, "rows", None)
    x = np.asarray(rows("token_embd.weight", tokens) if rows is not None else
                   W["token_embd.weight"][np.asarray(tokens)], np.float64)
    cos, sin = spec.rope(np.arange(T))
    r = spec.rot

    def rope(v):                                         # [T, heads, d]
        a, b = v[..., :r // 2], v[..., r // 2:r]
        c, s = cos[:, None, :], sin[:, None, :]
        return np.concatenate([a * c - b * s, b * c + a * s, v[..., r:]], axis=-1)

    for i in range(spec.layers):
        p = f"blk.{i}."
        h = _rms(x, W[p + "attn_norm.weight"], spec.eps)
        q = (h @ W[p + "attn_q.weight"].T).reshape(T, nq, d)
        kk = (h @ W[p + "attn_k.weight"].T).reshape(T, nkv, d)
        v = (h @ W[p + "attn_v.weight"].T).reshape(T, nkv, d)
        q = _rms(q, W[p + "attn_q_norm.weight"], spec.eps)
        kk = _rms(kk, W[p + "attn_k_norm.weight"], spec.eps)
        if spec.kind(i) == SLIDE:
            q, kk = rope(q), rope(kk)
        g = nq // nkv
        o = np.empty((T, nq, d))
        tq = np.arange(T)
        for hq in range(nq):
            s = q[:, hq] @ kk[:, hq // g].T / np.sqrt(d)
            bad = tq[None, :] > tq[:, None]
            if spec.kind(i) == SLIDE:
                bad |= tq[:, None] - tq[None, :] >= spec.window
            s = np.where(bad, -np.inf, s)
            s = np.exp(s - s.max(-1, keepdims=True))
            o[:, hq] = (s / s.sum(-1, keepdims=True)) @ v[:, hq // g]
        x = x + o.reshape(T, -1) @ W[p + "attn_output.weight"].T
        h = _rms(x, W[p + "ffn_norm.weight"], spec.eps)
        lg = h @ np.asarray(W[p + "ffn_gate_inp.weight"], np.float64).T          # [T, E]
        y = np.zeros_like(x)
        lim = spec.clamp
        for t in range(T):
            ids, wts = route(lg[t], spec.k)
            if routes is not None:
                pr = np.sort(np.exp(lg[t] - lg[t].max()))[::-1]
                routes.append((t, i, [int(e) for e in ids], pr[spec.k - 1] - pr[spec.k]))
            for e, wt in zip(ids, wts):
                wg = W.expert(p + "ffn_gate_exps.weight", int(e))
                wu = W.expert(p + "ffn_up_exps.weight", int(e))
                wd = W.expert(p + "ffn_down_exps.weight", int(e))
                gg, uu = wg @ h[t], wu @ h[t]
                gg = np.minimum(gg, lim)
                a = gg / (1 + np.exp(-gg)) * np.clip(uu, -lim, lim)
                y[t] += wt * (wd @ a)
        x = x + y
    x = _rms(x, W["output_norm.weight"], spec.eps)
    return (x.astype(np.float32) @ np.asarray(W["output.weight"], np.float32).T).astype(np.float64)


# =========================================================================== the device's form
class DeviceWeights(Mapping):
    """The GGUF's tensors under the names lfm2.Image reads (model.layers.i.self_attn.q_proj,
    q_layernorm, out_proj, feed_forward.gate / expert_bias / experts.e.w1 w3 w2, operator_norm,
    ffn_norm, embedding_norm, embed_tokens, lm_head), each q and k head's rows and norm gains in
    PERM's order; lazily."""

    def __init__(self, W, spec: Spec):
        self.W, self.spec = W, spec
        L, E = spec.layers, spec.E
        self.names = ["model.embed_tokens.weight", "lm_head.weight", "model.embedding_norm.weight"]
        for i in range(L):
            p = f"model.layers.{i}."
            self.names += [p + n for n in (
                "operator_norm.weight", "ffn_norm.weight", "self_attn.q_proj.weight",
                "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.out_proj.weight",
                "self_attn.q_layernorm.weight", "self_attn.k_layernorm.weight",
                "feed_forward.gate.weight", "feed_forward.expert_bias")]

    def _heads(self, w, n):
        d = self.spec.head_dim
        w = np.asarray(w, np.float32)
        return w.reshape(n, d, *w.shape[1:])[:, PERM].reshape(w.shape)

    def __getitem__(self, k):
        W, s = self.W, self.spec
        top = {"model.embed_tokens.weight": "token_embd.weight", "lm_head.weight": "output.weight",
               "model.embedding_norm.weight": "output_norm.weight"}
        if k in top:
            return np.asarray(W[top[k]], np.float32)
        if not k.startswith("model.layers."):
            raise KeyError(k)
        i, rest = k[len("model.layers."):].split(".", 1)
        p = f"blk.{i}."
        if rest.startswith("feed_forward.experts."):
            e, w = rest[len("feed_forward.experts."):].split(".", 1)
            n = {"w1.weight": "ffn_gate_exps", "w3.weight": "ffn_up_exps",
                 "w2.weight": "ffn_down_exps"}[w]
            return W.expert(p + n + ".weight", int(e))
        if rest == "feed_forward.expert_bias":
            return np.zeros(s.E, np.float32)
        m = {"operator_norm.weight": "attn_norm.weight", "ffn_norm.weight": "ffn_norm.weight",
             "self_attn.v_proj.weight": "attn_v.weight",
             "self_attn.out_proj.weight": "attn_output.weight",
             "feed_forward.gate.weight": "ffn_gate_inp.weight"}
        if rest in m:
            return np.asarray(W[p + m[rest]], np.float32)
        if rest == "self_attn.q_proj.weight":
            return self._heads(W[p + "attn_q.weight"], s.n_q)
        if rest == "self_attn.k_proj.weight":
            return self._heads(W[p + "attn_k.weight"], s.n_kv)
        if rest in ("self_attn.q_layernorm.weight", "self_attn.k_layernorm.weight"):
            g = np.asarray(W[p + ("attn_q_norm.weight" if rest[10] == "q" else
                                  "attn_k_norm.weight")], np.float32)
            return g[PERM]
        raise KeyError(k)

    def __contains__(self, k):
        try:
            self[k] if "experts" not in k else None
        except KeyError:
            return False
        return k in self.names or "feed_forward.experts." in k

    def __iter__(self):
        return iter(self.names)

    def __len__(self):
        return len(self.names)


@dataclass(frozen=True)
class DevSpec(L2.Spec):
    """lfm2.Spec of a Maple model (every layer attention + MoE) and its Spec (the kinds, RoPE,
    window)."""
    maple: Spec | None = None

    @property
    def akinds(self) -> tuple:
        return tuple(self.maple.kind(i) for i in range(self.layers))

    def check(self, cfg) -> None:
        super().check(cfg)
        if cfg.S != 1:
            raise ValueError("Maple runs on one slice (S = 1)")

    def image(self, cfg, cap: int, batch: int = 1, rows: int = 1, wformat: str = "int8",
              head_format=None, lookup=False, experts=None, embed_host=None,
              formats=None) -> "Image":
        if rows != 1:
            raise NotImplementedError("Maple: one token a run (Engine(rows=1))")
        return Image(self, cfg, cap, batch, rows, wformat, head_format, lookup, experts,
                     embed_host, formats)


class Image(L2.Image):
    """lfm2.Image plus Maple's RoPE tables of both kinds [cap, d/2] (Spec.dev_tables). A layer's
    formats group carries its kind: hardware loops run layers of one kind (as lfm3.Image)."""

    def _groups(self, lf):
        return tuple(g + (k,) for g, k in zip(lf, self._akinds))

    def _layout(self, g, mlp, rb):
        return super()._layout(g[:4], mlp, rb)

    def __init__(self, spec: DevSpec, cfg, cap, batch=1, rows=1, wformat="int8",
                 head_format=None, lookup=False, experts=None, embed_host=None, formats=None):
        if cap > spec.maple.window:
            raise NotImplementedError(
                f"Maple: a context of {cap} tokens, past the sliding window "
                f"({spec.maple.window}); at most the window is exact (sliding = full) here")
        self._akinds = spec.akinds
        super().__init__(spec, cfg, cap, batch, rows, wformat, head_format, lookup, experts,
                         embed_host, formats)
        d = spec.head_dim
        b = self.nbytes
        self.ropes = {}
        for k in (FULL, SLIDE):
            self.ropes[k] = (b, b + 4 * cap * (d // 2))
            b += 8 * cap * (d // 2)
        self.nbytes = (b + 4095) // 4096 * 4096
        if self.nbytes > cfg.DRAM_BYTES:
            raise MemoryError(f"model image needs {self.nbytes / 2**20:.0f} MiB, DRAM_BYTES is "
                              f"{cfg.DRAM_BYTES / 2**20:.0f} MiB")

    def build(self, W) -> list:
        spec = self.spec
        dw = W if isinstance(W, DeviceWeights) else DeviceWeights(W, spec.maple)
        imgs = super().build(dw)
        img, d = imgs[0], spec.head_dim

        def put(addr, a):
            v = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
            img[addr:addr + v.size] = v
        for k, (ca, sa) in self.ropes.items():
            c, s = spec.maple.dev_tables(k, np.arange(self.cap))
            put(ca, F.ftz(np.asarray(c, np.float32)))
            put(sa, F.ftz(np.asarray(s, np.float32)))
        return imgs

    def expert(self, W: dict, g: int) -> np.ndarray:
        dw = W if isinstance(W, DeviceWeights) else DeviceWeights(W, self.spec.maple)
        return super().expert(dw, g)

    def descriptors(self, sid: int) -> SimpleNamespace:
        m = super().descriptors(sid)
        spec, d, cap = self.spec, self.spec.head_dim, self.cap
        base_layer = m.layer

        def layer(li, it=None, at=None):
            lw = base_layer(li, it, at)
            lw.akind = spec.akinds[li]
            return lw

        m.layer = layer
        m.ropes = {k: (Tensor(Affine(c), (cap, d // 2), (d // 2, 1)),
                       Tensor(Affine(s), (cap, d // 2), (d // 2, 1)))
                   for k, (c, s) in self.ropes.items()}
        return m

    def compile_step(self, pos: int, block: int = ATTN_BLOCK, tok: int | None = None) -> list:
        return [trace_fit(maple_step, self.cfg, s,
                          lambda s=s: {"m": step_descriptors(self, s), "pos": pos,
                                       "block": block, **_tok_arg(self, tok)}).finish()
                for s in range(self.cfg.S)]

    def compile_rows(self, *a, **k):
        raise NotImplementedError("Maple: one token a run (compile_step)")

    def compile_decode(self, blocks: int, lo: int, block: int = ATTN_BLOCK):
        from .qwen3 import compile_decode
        return compile_decode(self, maple_step, blocks, lo, block)

    def compile_generate(self, blocks: int, lo: int, block: int = ATTN_BLOCK,
                         chain: bool = True, samp=None, debug: bool = False,
                         part: int | None = None) -> list:
        return G.compile_generate(self, maple_step, blocks, lo, block, chain, samp, debug, part)

    def compile_prompt_run(self, *a, **k):
        raise NotImplementedError("Maple: no prompt runs (one token a run)")


@ol.jit
def maple_step(m, pos: int, block: int = ATTN_BLOCK, tok: int | None = None):
    """One decode token: lfm2_step with the RoPE table of each layer's kind (full layers: no
    rotation) and every layer's MoE. Within the window a sliding layer attends to every earlier
    token (Image refuses a longer context)."""
    fill_logits(m)
    spec = m.spec
    x, _, _ = _inputs(m, pos, tok)
    pr = pos.pos if isinstance(pos, RunPos) else pos
    tabs = {k: (ol.load(c[pr, :]), ol.load(s[pr, :])) for k, (c, s) in m.ropes.items()}

    def layer(li, it):
        lw = m.layer(li, it)
        c, s_ = tabs[lw.akind]
        x.set(_attention(x, lw, c, s_, pos, spec, block))
        x.set(MO.moe_ffn(x, lw, spec.moe, m.moe_dev, spec.eps))

    L2.run_layers(m.plan, layer)
    _lm_head(x, m, spec)


def tiny_spec(**kw) -> Spec:
    """A small Maple for tests: 4 layers (the last full), H 256, 4 q heads of 128, 2 kv heads,
    16 experts of width 512, top 4, vocabulary 1000."""
    base = dict(hidden=256, layers=4, n_q=4, n_kv=2, full=(3,), E=16, k=4, ffn=512, vocab=1000,
                window=512, bos=0, eos=(1,))
    base.update(kw)
    return Spec(**base)


def tiny_weights(spec: Spec, seed: int = 0, o_scale: float = 0.03,
                 qk_gain: float = 1.0) -> DictWeights:
    """Random GGUF-named weights with TQ2_0's structure: attention and experts d * {-1, 0, +1}
    a block of 256 along a row (one d a block), fp32 router, norms near 1 (q / k norm gains
    times qk_gain: sharper attention, where RoPE's positions show in the logits)."""
    rng = np.random.default_rng(seed)
    H, d, E, F = spec.hidden, spec.head_dim, spec.E, spec.ffn

    def tern(rows, K, sc=0.05):
        t = rng.integers(-1, 2, (rows, K)).astype(np.float32)
        s = np.float16(sc) * (1 + 0.5 * rng.random((rows, K // 256))).astype(np.float16)
        return (t.reshape(rows, -1, 256) * s.astype(np.float32)[..., None]).reshape(rows, K)

    W = DictWeights()
    W["token_embd.weight"] = rng.normal(0, 1, (spec.vocab, H)).astype(np.float32)
    W["output.weight"] = rng.normal(0, 0.05, (spec.vocab, H)).astype(np.float32)
    W["output_norm.weight"] = (1 + 0.1 * rng.normal(size=H)).astype(np.float32)
    for i in range(spec.layers):
        p = f"blk.{i}."
        W[p + "attn_norm.weight"] = (1 + 0.1 * rng.normal(size=H)).astype(np.float32)
        W[p + "ffn_norm.weight"] = (1 + 0.1 * rng.normal(size=H)).astype(np.float32)
        W[p + "attn_q.weight"] = tern(spec.n_q * d, H)
        W[p + "attn_k.weight"] = tern(spec.n_kv * d, H)
        W[p + "attn_v.weight"] = tern(spec.n_kv * d, H)
        W[p + "attn_output.weight"] = tern(H, spec.n_q * d, o_scale)
        W[p + "attn_q_norm.weight"] = (qk_gain * (1 + 0.2 * rng.normal(size=d))).astype(np.float32)
        W[p + "attn_k_norm.weight"] = (qk_gain * (1 + 0.2 * rng.normal(size=d))).astype(np.float32)
        W[p + "ffn_gate_inp.weight"] = rng.normal(0, 0.1, (E, H)).astype(np.float32)
        W[p + "ffn_gate_exps.weight"] = np.stack([tern(F, H, 0.08) for _ in range(E)])
        W[p + "ffn_up_exps.weight"] = np.stack([tern(F, H, 0.08) for _ in range(E)])
        W[p + "ffn_down_exps.weight"] = np.stack([tern(H, F, 0.05) for _ in range(E)])
    return W


__all__ = ["Spec", "DevSpec", "Image", "GGUFWeights", "DictWeights", "DeviceWeights",
           "reference_logits", "route", "tq2_values", "tq2_ternary", "maple_step", "tiny_spec",
           "tiny_weights", "PERM", "SLIDE", "FULL", "replace", "Q", "Path"]
