"""NVIDIA Nemotron-Labs Dory 1.0 (GGUF architecture "dory": a looped normalized GPT, nGPT) on
the device graph. The math is llama.cpp's (brianchmielNV/llama.cpp 98a74c97, src/models/
dory.cpp: read, not executed):

    l2(x) = x / |x|  (rms_norm(eps / n) / sqrt(n))
    blocks: the physical pattern S-S-S-*-... -- an attention block (S: sliding window, *:
    full; RoPE theta per the block's profile: 2 -> rope.freq_base_2, 1 -> rope.freq_base),
    then an MLP block ("-"); a pair of them is one layer here.
    attention:  q, k, v = W x  (x as it is, no input norm); RoPE (rotate-half)
                q = l2(q_head) * (sqk sqrt(d) / init_std)^2;  k = l2(k_head)
                g = sigmoid(Wg x * (sgate sqrt(gate_init) / init_std)^2)
                y = Wo l2(attn(q, k, v) * g)                        (scale 1 / sqrt(d))
    MLP:        g = Wg x * suv_g sqrt(H);  u = Wu x * suv_u sqrt(H)
                y = Wd l2(silu(g) * u)
    residual:   x = l2(a + |alpha alpha_init / init_std| (l2(y) - a)),  a = l2(x)
    loops:      input layers; h = x; per loop: x = l2(x + h), the recurrent layers (their own
                KV per loop: per_loop); output layers; logits = Wout x * logit_scale / init_std

On the device (qwen3's kernels): the per-row scales fold into the weight rows (Wg, Wu, the
gate rows; the head's rows by logit_scale / init_std), the l2 before Wo and Wd drop out (a
scalar per row, normalized by the l2 after), q / k norms are qwen3's with gain 1 / sqrt(d)
and the query scale after RoPE (lw.qscale), l2(y) is qwen3's `post` with gain
alpha / sqrt(H) (x + alpha l2(y)), then x = l2(z - alpha x): the residual update. The image
is qwen3.Image's (a layer: the pair's attention and MLP), plus a table per pair (alphas,
posts, query scale), the gate rows, two RoPE tables and the KV areas of loops 1.. for the
recurrent layers. KV capacity at most the sliding window: every S layer sees all positions.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .. import fp32 as F
from .. import language as ol
from .. import qcache as QC
from .. import quant as Q
from ..compiler import Affine, KVDesc, QTensor, Tensor, trace_fit
from ..kernels.lib import rmsnorm
from . import qwen3 as Q3
from .qwen3 import (ATTN_BLOCK, RunPos, _attention, _lm_head, _mlp, _tok_arg, _embed,
                    fill_logits, step_descriptors)

GROUP = 4                       # the pattern's period in layers (S, S, S, *): a loop body


@dataclass(frozen=True)
class Spec(Q3.Spec):
    n_input: int = 8            # layers (attention + MLP pairs)
    n_rec: int = 20
    n_out: int = 8
    loops: int = 4
    theta2: float = 1e4
    kinds: tuple = ()           # per layer: its RoPE profile (1: theta, 2: theta2)
    init_std: float = 0.0177
    alpha_init: float = 0.05
    gate_init: float = 50.6
    window: int = 1024

    def image(self, cfg, cap: int, batch: int = 1, rows: int = 1, wformat: str = "int8",
              head_format=None, lookup: bool = False, embed_host=None, formats=None,
              steering: bool = False, kv=None) -> "Image":
        return Image(self, cfg, cap, batch, rows, wformat, head_format, lookup, embed_host,
                     formats, steering, kv)


def spec_from_gguf(g, loops: int | None = None) -> Spec:
    kv = g.kv
    pat = kv["dory.layer_pattern"]
    n_i, n_r, n_o = (kv[f"dory.{k}_layer_count"] for k in ("input", "recurrent", "output"))
    prof = list(kv["dory.rope.profile"])
    if len(pat) != n_i + n_r + n_o or pat[1::2] != "-" * (len(pat) // 2) or \
            set(pat[0::2]) - set("S*") or n_i % 2 or n_r % 2 or n_o % 2:
        raise ValueError("Dory: attention / MLP pairs expected (S-, *-)")
    kinds = tuple(prof[0::2])
    if any(k not in (1, 2) for k in kinds) or any((k == 2) != (c == "S")
                                                 for k, c in zip(kinds, pat[0::2])):
        raise ValueError("Dory: the RoPE profile is not the pattern's (S: 2, *: 1)")
    L = len(pat) // 2
    if kinds != tuple(kinds[i % GROUP] for i in range(L)) or \
            any(n % (2 * GROUP) for n in (n_i, n_r, n_o)):
        raise ValueError(f"Dory: the pattern repeats every {GROUP} layers in each part")
    if kv.get("dory.recurrent_kv_cache_mode", "per_loop") != "per_loop":
        raise ValueError("Dory: per_loop KV only")
    H, nh, nk = kv["dory.embedding_length"], kv["dory.attention.head_count"], \
        kv["dory.attention.head_count_kv"]
    d = kv["dory.attention.key_length"]
    if kv["dory.rope.dimension_count"] != d or kv["dory.attention.value_length"] != d:
        raise ValueError("Dory: full-head RoPE, equal K / V heads")
    return Spec(hidden=H, layers=L, n_q=nh, n_kv=nk, head_dim=d,
                ffn=kv["dory.feed_forward_length"], vocab=g.tensors["output.weight"].shape[1],
                eps=1e-12, theta=kv["dory.rope.freq_base"], tied=False,
                bos=kv.get("tokenizer.ggml.bos_token_id", 1),
                eos=(kv.get("tokenizer.ggml.eos_token_id", 11),), qk_norm=True,
                qk_smooth=False, n_input=n_i // 2, n_rec=n_r // 2, n_out=n_o // 2,
                loops=int(loops or kv["dory.num_loops"]), theta2=kv["dory.rope.freq_base_2"],
                kinds=kinds, init_std=kv["dory.init_std"], alpha_init=kv["dory.alpha_init"],
                gate_init=kv["dory.gate_init"], window=kv["dory.attention.sliding_window"])


class Raw(Mapping):
    """The checkpoint's tensors by GGUF name, fp32 (dequantized as read), [n, k] for matrices."""

    def __init__(self, path):
        from .. import gguf as GG
        self.g = GG.GGUF(path)
        self._small = {}

    def __getitem__(self, name):
        from .. import gguf as GG
        if name in self._small:
            return self._small[name]
        ti = self.g.tensors[name]
        v = GG.dequant(self.g.raw(name), ti.type).reshape(ti.shape[::-1])
        if v.size <= 1 << 16:
            self._small[name] = v
        return v

    def rows(self, name, ids):
        from .. import gguf as GG
        ti = self.g.tensors[name]
        k = ti.shape[0]
        be, bb = GG.BLOCK[ti.type]
        raw = self.g.raw(name).reshape(-1, k // be * bb)[np.asarray(ids)].reshape(-1)
        return GG.dequant(raw, ti.type).reshape(len(ids), k)

    def __iter__(self):
        return iter(self.g.tensors)

    def __len__(self):
        return len(self.g.tensors)


def _scales(spec, R, P):
    """Layer P's folded scales: q (per query element), gate rows, g / u rows, the alphas."""
    a, m = f"blk.{2 * P}.", f"blk.{2 * P + 1}."
    d, H, s = spec.head_dim, spec.hidden, spec.init_std
    f = lambda n: np.asarray(R[n], np.float64)                       # noqa: E731
    return dict(q=(f(a + "attn_qk_scale.weight") * math.sqrt(d) / s) ** 2,
                gate=(f(a + "attn_gate_scale.weight") * math.sqrt(spec.gate_init) / s) ** 2,
                g=f(m + "ffn_gate_scale.weight") * math.sqrt(H),
                u=f(m + "ffn_up_scale.weight") * math.sqrt(H),
                alpha_a=np.abs(f(a + "residual_alpha.weight") * spec.alpha_init / s),
                alpha_m=np.abs(f(m + "residual_alpha.weight") * spec.alpha_init / s))


def schedule(spec):
    """(physical layer, loop) in run order: the input layers, the recurrent ones per loop, the
    output layers (loop 0 outside the recurrence)."""
    ni, nr = spec.n_input, spec.n_rec
    return ([(P, 0) for P in range(ni)] +
            [(ni + P, k) for k in range(spec.loops) for P in range(nr)] +
            [(ni + nr + P, 0) for P in range(spec.n_out)])


def reference_logits(spec: Spec, R, tokens) -> np.ndarray:
    """fp32 numpy forward of the sequence (causal), dory.cpp's graph on the raw tensors."""
    tokens = list(tokens)
    T, d, G, H = len(tokens), spec.head_dim, spec.n_q // spec.n_kv, spec.hidden

    def l2(v):
        return v / np.sqrt(np.sum(v * v, axis=-1, keepdims=True) + 1e-6)

    def tables(theta):
        inv = 1.0 / (theta ** (np.arange(d // 2, dtype=np.float64) * 2 / d))
        ang = np.arange(T)[:, None] * inv
        return np.cos(ang)[:, None, :].astype(np.float32), np.sin(ang)[:, None, :].astype(np.float32)

    tabs = {1: tables(spec.theta), 2: tables(spec.theta2)}

    def rot(v, k):
        c, s_ = tabs[k]
        h = d // 2
        return np.concatenate([v[..., :h] * c - v[..., h:] * s_, v[..., h:] * c + v[..., :h] * s_], -1)

    mask = np.triu(np.full((T, T), -np.inf, np.float32), 1)
    x = R.rows("token_embd.weight", tokens).astype(np.float32)
    h_in = None
    for P, k in schedule(spec):
        if P == spec.n_input and k == 0 and h_in is None:
            h_in = x
        if spec.n_input <= P < spec.n_input + spec.n_rec and P == spec.n_input:
            x = l2(x + h_in)
        sc = _scales(spec, R, P)
        a = f"blk.{2 * P}."
        q = (x @ R[a + "attn_q.weight"].T).reshape(T, spec.n_q, d)
        kk = (x @ R[a + "attn_k.weight"].T).reshape(T, spec.n_kv, d)
        v = (x @ R[a + "attn_v.weight"].T).reshape(T, spec.n_kv, d)
        q = l2(rot(q, spec.kinds[P])) * sc["q"].reshape(spec.n_q, d)
        kk = l2(rot(kk, spec.kinds[P]))
        gt = 1 / (1 + np.exp(-(x @ R[a + "attn_gate.weight"].T) * sc["gate"]))
        o = np.zeros((T, spec.n_q, d), np.float64)
        for hq in range(spec.n_q):
            s = q[:, hq] @ kk[:, hq // G].T / math.sqrt(d) + mask
            s = np.exp(s - s.max(axis=1, keepdims=True))
            o[:, hq] = (s / s.sum(axis=1, keepdims=True)) @ v[:, hq // G]
        y = l2(o.reshape(T, -1) * gt) @ R[a + "attn_output.weight"].T
        r = l2(x)
        x = l2(r + sc["alpha_a"] * (l2(y) - r)).astype(np.float32)
        m = f"blk.{2 * P + 1}."
        g = (x @ R[m + "ffn_gate.weight"].T) * sc["g"]
        u = (x @ R[m + "ffn_up.weight"].T) * sc["u"]
        y = l2((g / (1 + np.exp(-g))) * u) @ R[m + "ffn_down.weight"].T
        r = l2(x)
        x = l2(r + sc["alpha_m"] * (l2(y) - r)).astype(np.float32)
    ls = np.asarray(R["output_scale.weight"], np.float64) / spec.init_std
    out = np.empty((T, spec.vocab), np.float32)
    for r0 in range(0, spec.vocab, 8192):
        ids = np.arange(r0, min(spec.vocab, r0 + 8192))
        out[:, ids] = (x @ R.rows("output.weight", ids).T) * ls[ids]
    return out


class Weights(Mapping):
    """Raw tensors under qwen3's names (layer P: the pair's attention and MLP), the scales folded
    into the rows (gate / up, the head's rows), the norms qwen3's gains for an l2."""

    def __init__(self, spec: Spec, R: Raw, vocab: int | None = None):
        self.spec, self.R = spec, R
        self.V = vocab or spec.vocab
        H, d = spec.hidden, spec.head_dim
        self._where = {}
        for P in range(spec.layers):
            p, a, m = f"model.layers.{P}.", f"blk.{2 * P}.", f"blk.{2 * P + 1}."
            for hf, gg in (("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"),
                           ("self_attn.v_proj", "attn_v"), ("self_attn.o_proj", "attn_output")):
                self._where[p + hf + ".weight"] = ("raw", a + gg + ".weight")
            self._where[p + "mlp.down_proj.weight"] = ("raw", m + "ffn_down.weight")
            self._where[p + "mlp.gate_proj.weight"] = ("rows", m + "ffn_gate.weight", P, "g")
            self._where[p + "mlp.up_proj.weight"] = ("rows", m + "ffn_up.weight", P, "u")
            self._where[p + "self_attn.gate_proj.weight"] = ("rows", a + "attn_gate.weight", P,
                                                              "gate")
            for n in ("input_layernorm", "post_attention_layernorm"):
                self._where[p + n + ".weight"] = ("const", H, 1 / math.sqrt(H))
            for n in ("q_norm", "k_norm"):
                self._where[p + f"self_attn.{n}.weight"] = ("const", d, 1 / math.sqrt(d))
        self._where["model.norm.weight"] = ("const", H, 1 / math.sqrt(H))
        self._where["model.embed_tokens.weight"] = ("embed",)
        self._where["lm_head.weight"] = ("head",)
        self._sc = {}

    def scales(self, P):
        if P not in self._sc:
            self._sc = {P: _scales(self.spec, self.R, P)}
        return self._sc[P]

    def __getitem__(self, name):
        w = self._where[name]
        if w[0] == "raw":
            return self.R[w[1]]
        if w[0] == "rows":
            return (self.R[w[1]] * self.scales(w[2])[w[3]][:, None]).astype(np.float32)
        if w[0] == "const":
            return np.full(w[1], w[2], np.float32)
        if w[0] == "embed":
            return _Rows(self.R, "token_embd.weight", self.V, self.spec.hidden)
        ls = (np.asarray(self.R["output_scale.weight"], np.float64) / self.spec.init_std)[:self.V]
        return (self.R.rows("output.weight", np.arange(self.V)) * ls[:, None]).astype(np.float32)

    def embedding_table(self):
        return self.R.rows("token_embd.weight", np.arange(self.V)).astype(np.float32)

    def __iter__(self):
        return iter(self._where)

    def __len__(self):
        return len(self._where)


class _Rows:
    """A GGUF matrix's rows on demand ([ids] / [a:b])."""

    def __init__(self, R, name, n, k):
        self.R, self.name, self.shape = R, name, (n, k)

    def __getitem__(self, ids):
        if isinstance(ids, slice):
            ids = range(*ids.indices(self.shape[0]))
        r = self.R.rows(self.name, np.atleast_1d(np.asarray(ids))).astype(np.float32)
        return r[0] if np.ndim(ids) == 0 else r


class Image(Q3.Image):
    """qwen3.Image (a layer: a pair) plus per layer the gate rows (the attention format) and a
    table [alpha_a, post_a, alpha_m, post_m (H each), qscale (n_q d)], two RoPE tables
    [cap, d/2] (profiles 1, 2) and the KV areas of loops 1.. for the recurrent layers."""

    def __init__(self, spec: Spec, cfg, cap, batch=1, rows=1, wformat="int8", head_format=None,
                 lookup=False, embed_host=None, formats=None, steering=False, kv=None):
        if cap > spec.window:
            raise ValueError(f"Dory: KV capacity {cap} above the sliding window ({spec.window})")
        if cfg.S != 1 or batch != 1 or steering:
            raise ValueError("Dory: one slice, one sequence")
        super().__init__(spec, cfg, cap, batch, rows, wformat, head_format, lookup, embed_host,
                         formats, steering, kv)
        if len(set(self.lf)) != 1:
            raise ValueError("Dory: one weight format for every layer")
        H, d, L, D = spec.hidden, spec.head_dim, spec.layers, cfg.D
        nq = spec.n_q * d
        b = (self.nbytes + 4095) // 4096 * 4096
        self.gfmt = self.lf[0][0]
        self.g_rb = nq * Q.row_bytes(H, self.gfmt, D)
        self.g_sb = nq * Q.scale_bytes(H, self.gfmt, D)
        self.gstride = (self.g_rb + self.g_sb + 4095) // 4096 * 4096
        self.gates = b
        b += L * self.gstride
        self.tstride = 4 * (4 * H + nq)
        self.tab = b
        b += L * self.tstride
        b = (b + 4095) // 4096 * 4096
        self.ropes = {}
        for k in (1, 2):
            self.ropes[k] = (b, b + 4 * cap * (d // 2))
            b += 8 * cap * (d // 2)
        b = (b + 4095) // 4096 * 4096
        offs = [v for heads in self.lofs["kvs"] for r in heads for v in r.values()]
        self.kv_lo = min(offs)
        self.kv_span = (self.LS - self.kv_lo + 4095) // 4096 * 4096
        self.kv_area = b                # every loop's recurrent KV (a loop: n_rec spans)
        b += spec.loops * spec.n_rec * self.kv_span
        self.nbytes = (b + 4095) // 4096 * 4096
        if self.nbytes > cfg.DRAM_BYTES:
            raise MemoryError(f"model image needs {self.nbytes / 2**20:.0f} MiB, DRAM_BYTES is "
                              f"{cfg.DRAM_BYTES / 2**20:.0f} MiB")

    def build(self, W) -> list:
        imgs = super().build(W)
        img = imgs[0]
        spec, H, d, D = self.spec, self.spec.hidden, self.spec.head_dim, self.cfg.D

        def put(addr, a):
            v = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
            img[addr:addr + v.size] = v

        qb = QC.Batch()
        for P in range(spec.layers):
            sc = W.scales(P)
            g = W[f"model.layers.{P}.self_attn.gate_proj.weight"]
            o = self.gates + P * self.gstride
            qb.submit(g, self.gfmt, D, lambda q, s, o=o: (put(o, q), put(o + self.g_rb, s)))
            t = np.concatenate([sc["alpha_a"], sc["alpha_a"] / math.sqrt(H), sc["alpha_m"],
                                sc["alpha_m"] / math.sqrt(H), sc["q"]])
            put(self.tab + P * self.tstride, F.ftz(t.astype(np.float32)))
        qb.wait()
        for k, (ca, sa) in self.ropes.items():
            th = spec.theta if k == 1 else spec.theta2
            inv = 1.0 / (th ** (np.arange(d // 2, dtype=np.float64) * 2 / d))
            ang = np.arange(self.cap)[:, None] * inv
            put(ca, np.cos(ang).astype(np.float32))
            put(sa, np.sin(ang).astype(np.float32))
        return imgs

    def descriptors(self, sid: int) -> SimpleNamespace:
        m = super().descriptors(sid)
        spec, cfg, H, d, D = self.spec, self.cfg, self.spec.hidden, self.spec.head_dim, self.cfg.D
        nq = spec.n_q * d
        base_layer = m.layer

        def layer(P, loop=None):
            """Layer P (an int, or an Affine of loop variables); a recurrent one at `loop` (an
            int or a loop variable: its KV in the loops' area), else its block's KV."""
            idx = Affine.of(P)
            lw = base_layer(0, idx)
            lw.g_in = None
            t = Affine(self.tab) + idx * self.tstride
            lw.alpha_a = Tensor(t, (H,), (1,))
            lw.post_a = Tensor(t + 4 * H, (H,), (1,))
            lw.alpha_m = Tensor(t + 8 * H, (H,), (1,))
            lw.post_m = Tensor(t + 12 * H, (H,), (1,))
            lw.qscale = Tensor(t + 16 * H, (nq,), (1,))
            go = Affine(self.gates) + idx * self.gstride
            lw.wgate = QTensor.weight(go, go + self.g_rb, nq, H, self.gfmt, D)
            if loop is not None:
                off = Affine(self.kv_area - self.kv_lo) + \
                    Affine.of(loop) * (spec.n_rec * self.kv_span) + \
                    (idx - spec.n_input) * self.kv_span
                lw.kvs = [KVDesc({sid + j * cfg.S: {k: off + v for k, v in r.items()}
                                  for j, r in enumerate(heads)}, self.cap, self.dk, D, cfg.S,
                                 sid, dv=d, kfmt="int8" if self.kv == "int8" else "kv6",
                                 vfmt="kv6" if self.kv == "k6v6" else "int8",
                                 vones=self.io.get("v6ones"), vplanes=self.v6planes)
                          for heads in self.lofs["kvs"]]
                lw.kv = lw.kvs[0]
            return lw

        m.layer = layer
        m.ropes = {k: (Tensor(Affine(c), (self.cap, d // 2), (d // 2, 1)),
                       Tensor(Affine(s), (self.cap, d // 2), (d // 2, 1)))
                   for k, (c, s) in self.ropes.items()}
        return m

    def compile_step(self, pos: int, block: int = ATTN_BLOCK, tok: int | None = None) -> list:
        return [trace_fit(dory_step, self.cfg, s,
                          lambda s=s: {"m": step_descriptors(self, s), "pos": pos,
                                       "block": block, **_tok_arg(self, tok)}).finish()
                for s in range(self.cfg.S)]

    def compile_rows(self, *a, **k):
        raise NotImplementedError("Dory: one token a run (compile_step)")

    def compile_decode(self, blocks: int, lo: int, block: int = ATTN_BLOCK):
        return Q3.compile_decode(self, dory_step, blocks, lo, block)

    def compile_generate(self, blocks: int, lo: int, block: int = ATTN_BLOCK,
                         chain: bool = True, samp=None, debug: bool = False,
                         part: int | None = None) -> list:
        from . import generate as G
        return G.compile_generate(self, dory_step, blocks, lo, block, chain, samp, debug, part)


def _l2(x, eps):
    return rmsnorm(x, None, eps) * (1.0 / math.sqrt(x.cols))


@ol.jit
def dory_step(m, pos: int, block: int = ATTN_BLOCK, tok: int | None = None):
    """One decode token at position `pos`: its embedding -> logits (dory.cpp's graph; the layers
    as hardware loops over groups of GROUP, each body the group's RoPE kinds)."""
    fill_logits(m)
    spec = m.spec
    eps = spec.eps
    if isinstance(pos, RunPos):
        x = _embed(m, pos.tok)
        ol.release(pos.tok)
        pr = pos.pos
    else:
        x = ol.load(m.x) if tok is None else _embed(m, tok)
        pr = pos
    tabs = {k: (ol.load(c[pr, :]), ol.load(s[pr, :])) for k, (c, s) in m.ropes.items()}

    def run(first, n, loop=None):
        for it in ol.range(n // GROUP):
            for e in range(GROUP):
                lw = m.layer(Affine.of(it) * GROUP + (first + e), loop)
                c, s_ = tabs[spec.kinds[first + e]]
                # x + alpha l2(y) (qwen3's post) -> l2(a + alpha (l2(y) - a)), a = l2(x): x
                # itself but for the first layer's input (the raw embedding)
                a = _l2(x, eps)
                z = _attention(x, lw, c, s_, pos, spec, block, gated=True, post=lw.post_a)
                x.set(_l2(z - x + a - ol.load(lw.alpha_a)[None, :] * a, eps))
                z = _mlp(x, lw, spec, post=lw.post_m)
                x.set(_l2(z - ol.load(lw.alpha_m)[None, :] * x, eps))

    run(0, spec.n_input)
    h = x * 1.0
    for k in ol.range(spec.loops):
        x.set(_l2(x + h, eps))
        run(spec.n_input, spec.n_rec, k)
    run(spec.n_input + spec.n_rec, spec.n_out)
    _lm_head(x, m, spec)
