"""Cactus Needle (opentpu/llm/needle.py's model) on the device: every operation of the
encoder-decoder runs in device programs, resident in one DRAM image; the host writes the
image once and reads the logits.

    encoder program (the prompt's T rows at once, its ids compile-time values): the
        embedding rows from the fp32 table, 12 layers of bidirectional self attention
        (qwen3._attention_rows: every row's K / V appended, then each row attends all T; the
        ZCRMSNorm gains 1 + w; q / k head norms; RoPE), the final norm, then each decoder
        layer's cross K / V of the encoder rows (k_norm, no RoPE: identity tables) into its
        cross cache.
    decoder step (a token at a position): its embedding row; 8 layers of causal self
        attention (qwen3._attention, the KV cache) and cross attention (cross: the query alone,
        attending the T encoder tokens); the final norm and the tied head (int8 rows) ->
        logits [V].

The scalar residual gates sigmoid(g) fold into W_o's rows (x + s W_o a = x + (s W_o) a).
Weights int8 (MX rows), activations int8 at each projection, as the qwen3 path.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from .. import fp32 as F
from .. import language as ol
from .. import quant as Q
from ..compiler import Affine, KVDesc, QTensor, Tensor, k_scale_bytes, trace_fit
from ..kernels.lib import rmsnorm
from . import needle as N
from . import qwen3 as Q3
from .qwen3 import ATTN_BLOCK, _attention, _attention_rows

SPEC = Q3.Spec(hidden=N.D, layers=N.DEC_LAYERS, n_q=N.NH, n_kv=N.NK, head_dim=N.HD,
               ffn=N.D, vocab=8192, eps=N.EPS, theta=N.THETA, tied=True, bos=2, eos=(1,),
               qk_norm=True, qk_smooth=False)


class _Bump:
    def __init__(self, base=0):
        self.next = base

    def alloc(self, n, align=64):
        a = -(-self.next // align) * align
        self.next = a + n
        return a


class Image:
    """The DRAM image: per layer a block (encoder: its self attention; decoder: self and cross
    attention), the blocks of a stack at one stride (a hardware loop over them), the norms,
    the RoPE and identity tables, the embedding (fp32) and the tied head (int8), the I/O."""

    def __init__(self, cfg, W: dict, enc_cap: int = 128, dec_cap: int = 128,
                 wformat: str = "int8", cq: dict | None = None):
        """cq (needle.cq_matrices: the CACT CQ4 export): the attention matrices as TMM CQ
        codes (the int8 LUT of the shared codebook, fp32 group scales) with their activations
        in the CQ basis (the VPU's Hadamard); wformat then the head's (the embedding)."""
        if cfg.S != 1:
            raise ValueError("Needle: one slice")
        D = cfg.D
        self.cfg, self.D, self.enc_cap, self.dec_cap, self.fmt = cfg, D, enc_cap, dec_cap, wformat
        H, d = N.D, N.HD
        self.V = V = W["model.embed_tokens.weight"].shape[0]
        self.dk = dk = -(-d // D) * D
        nq, nk = N.NH * d, N.NK * d
        rb, sb = (lambda k: Q.row_bytes(k, wformat, D)), (lambda k: Q.scale_bytes(k, wformat, D))
        self.cq, self.lut = None, None
        if cq:
            from ..cactus.transcode import cq_to_tmm_cq
            self.cq = {n: cq_to_tmm_cq(m) for n, m in cq.items()}
            f0, p0, _, self.lut = next(iter(self.cq.values()))
            if any(not np.array_equal(t[3], self.lut) or
                   not np.array_equal(t[0].left_signs, f0.left_signs) or
                   not np.array_equal(t[0].right_signs, f0.right_signs) or
                   not np.array_equal(t[0].input_scale_recip, f0.input_scale_recip)
                   for t in self.cq.values()):
                raise ValueError("Needle CQ: one LUT and one activation basis expected")
            self.cq_bits = f0.bits
            rb = lambda k: -(-(k // D) // (4 if f0.bits == 2 else 2)) * D       # noqa: E731
            sb = lambda k: 4 * (k // D)                                       # noqa: E731

        def attn_layout(lb, cap):
            L = {"g_in": lb.alloc(4 * H), "qn": lb.alloc(4 * d), "kn": lb.alloc(4 * d)}
            for name, n in (("wq", nq), ("wk", nk), ("wv", nk), ("wo", H)):
                k = nq if name == "wo" else H
                L[name] = (lb.alloc(n * rb(k)), lb.alloc(n * sb(k)), n, k)
            L["kv"] = [{"k": lb.alloc(cap * dk), "ks": lb.alloc(cap * k_scale_bytes(dk, D)),
                        "vt": lb.alloc(dk * cap), "vs": lb.alloc(4 * cap)} for _ in range(N.NK)]
            return L

        lb = _Bump()
        self.enc_l = attn_layout(lb, enc_cap)
        self.enc_bs = -(-lb.next // 4096) * 4096
        lb = _Bump()
        self.dec_l = (attn_layout(lb, dec_cap), attn_layout(lb, enc_cap))
        self.dec_bs = -(-lb.next // 4096) * 4096
        b = _Bump(4096)
        self.enc0 = b.alloc(N.ENC_LAYERS * self.enc_bs, 4096)
        self.dec0 = b.alloc(N.DEC_LAYERS * self.dec_bs, 4096)
        self.puts = []
        f32 = lambda a: F.ftz(np.asarray(a, np.float32))            # noqa: E731
        one = lambda w: 1 + np.asarray(w, np.float64)                  # noqa: E731
        sig = lambda g: 1 / (1 + math.exp(-float(np.asarray(g).reshape(-1)[0])))  # noqa: E731

        def put_attn(base, L, p, gain, gate):
            self.puts += [(base + L["g_in"], f32(one(W[gain]))),
                          (base + L["qn"], f32(one(W[p + "q_norm.weight"]))),
                          (base + L["kn"], f32(one(W[p + "k_norm.weight"])))]
            for name, src in (("wq", "q_proj"), ("wk", "k_proj"), ("wv", "v_proj"),
                              ("wo", "out_proj")):
                if self.cq is not None:            # the codes; the gate in W_o's scales
                    _, q, sc, _ = self.cq[p + src + ".weight"]
                    if name == "wo":
                        sc = (sig(W[gate]) * sc).astype(np.float32)
                    if q.shape != (L[name][2], rb(L[name][3])):
                        raise ValueError(f"{p + src}: packed codes of an unexpected size")
                    self.puts += [(base + L[name][0], q), (base + L[name][1], sc)]
                    continue
                w = np.asarray(W[p + src + ".weight"], np.float64)
                if name == "wo":
                    w = sig(W[gate]) * w
                q, sc = Q.quantize_mxu(w.astype(np.float32), wformat, D)
                self.puts += [(base + L[name][0], q), (base + L[name][1], sc)]

        for i in range(N.ENC_LAYERS):
            p = f"model.encoder.layers.{i}."
            put_attn(self.enc0 + i * self.enc_bs, self.enc_l, p + "self_attn.",
                     p + "input_layernorm.weight", p + "attn_gate")
        for i in range(N.DEC_LAYERS):
            p, base = f"model.decoder.layers.{i}.", self.dec0 + i * self.dec_bs
            put_attn(base, self.dec_l[0], p + "self_attn.", p + "input_layernorm.weight",
                     p + "self_attn_gate")
            put_attn(base, self.dec_l[1], p + "encoder_attn.",
                     p + "encoder_attn_layer_norm.weight", p + "cross_attn_gate")

        def vec(a):
            a = f32(a)
            addr = b.alloc(a.nbytes)
            self.puts.append((addr, a))
            return Tensor(Affine(addr), (a.size,), (1,))

        def table(a):
            a = f32(a)
            addr = b.alloc(a.nbytes)
            self.puts.append((addr, a))
            return Tensor(Affine(addr), a.shape, (a.shape[1], 1))

        if self.cq is not None:                    # the shared activation basis
            f0 = next(iter(self.cq.values()))[0]
            self.cq_t = SimpleNamespace(
                recip=vec(np.tile(f0.input_scale_recip, 1)),
                left=vec(np.tile(f0.left_signs, H // 128)),
                right=vec(np.tile(f0.right_signs, H // 128)))
        self.enc_norm = vec(one(W["model.encoder.final_norm.weight"]))
        self.dec_norm = vec(one(W["model.decoder.norm.weight"]))
        E = np.asarray(W["model.embed_tokens.weight"], np.float32)
        self.embed = table(E)
        q, sc = Q.quantize_mxu(E, wformat, D)
        a0, a1 = b.alloc(q.nbytes), b.alloc(sc.nbytes)
        self.puts += [(a0, q), (a1, sc)]
        self.head = QTensor.weight(a0, a1, V, H, wformat, D)
        cap = max(enc_cap, dec_cap)
        inv = 1.0 / (N.THETA ** (np.arange(0, d, 2) / d))
        ang = np.arange(cap)[:, None] * inv
        self.cos, self.sin = table(np.cos(ang)), table(np.sin(ang))
        self.one_c, self.one_s = table(np.ones_like(ang)), table(np.zeros_like(ang))
        self.logits = Tensor(Affine(b.alloc(4 * V)), (1, V), (V, 1))
        self.nbytes = -(-b.next // 4096) * 4096
        if self.nbytes > cfg.DRAM_BYTES:
            raise MemoryError(f"Needle image {self.nbytes} bytes > DRAM_BYTES")

    def attn(self, L, off, cap, gain: bool = True):
        """An attention block's descriptors at block address `off` (an Affine)."""
        t = lambda a, n: Tensor(off + a, (n,), (1,))                       # noqa: E731
        lw = SimpleNamespace(g_in=t(L["g_in"], N.D) if gain else None, qn=t(L["qn"], N.HD),
                             kn=t(L["kn"], N.HD), rg=None, qscale=None)
        for name in ("wq", "wk", "wv", "wo"):
            a0, a1, n, k = L[name]
            if self.cq is not None:
                setattr(lw, name, QTensor.cq_weight(off + a0, off + a1, n, k, self.D,
                                                    self.cq_bits, self.lut))
            else:
                setattr(lw, name, QTensor.weight(off + a0, off + a1, n, k, self.fmt, self.D))
        lw.cq = self.cq_t if self.cq is not None else None
        lw.kv = KVDesc({j: {k: off + v for k, v in r.items()} for j, r in enumerate(L["kv"])},
                       cap, self.dk, self.D, 1, 0, dv=N.HD)
        lw.kvs = [lw.kv]
        return lw

    def enc_layer(self, it):
        return self.attn(self.enc_l, Affine(self.enc0) + Affine.of(it) * self.enc_bs,
                         self.enc_cap)

    def dec_layer(self, it, fill: bool = False):
        off = Affine(self.dec0) + Affine.of(it) * self.dec_bs
        return (self.attn(self.dec_l[0], off, self.dec_cap),
                self.attn(self.dec_l[1], off, self.enc_cap, gain=not fill))

    def build(self) -> np.ndarray:
        img = np.zeros(self.nbytes, np.uint8)
        for addr, a in self.puts:
            v = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
            img[addr:addr + v.size] = v
        return img


@ol.jit
def needle_encode(m, ids: tuple, block: int = ATTN_BLOCK):
    """The encoder over the prompt `ids` (T rows), then every decoder layer's cross K / V."""
    T, H = len(ids), N.D
    x = ol.empty([T, H], dense=True)
    for r, t in enumerate(ids):
        ol.load(m.embed[t:t + 1, :], out=x[r:r + 1, :])
    c = ol.empty([T, N.HD // 2], dense=True)
    s_ = ol.empty([T, N.HD // 2], dense=True)
    ol.load(m.cos[0:T, :], out=c)
    ol.load(m.sin[0:T, :], out=s_)
    rows = [(0, p) for p in range(T)]
    for it in ol.range(N.ENC_LAYERS):
        x.set(_attention_rows(x, m.enc_layer(it), c, s_, rows, SPEC, block, seq_all=T))
    x.set(rmsnorm(x, ol.load(m.enc_norm), N.EPS))
    ol.load(m.one_c[0:T, :], out=c)
    ol.load(m.one_s[0:T, :], out=s_)
    for it in ol.range(N.DEC_LAYERS):
        _attention_rows(x, m.dec_layer(it, fill=True)[1], c, s_, rows, SPEC, block,
                        kv_only=True)


@ol.jit
def needle_step(m, pos: int, tok: int, T: int, block: int = ATTN_BLOCK):
    """Decoder token `tok` at position `pos` over T encoder tokens -> m.logits."""
    x = ol.load(m.embed[tok:tok + 1, :])
    c, s_ = ol.load(m.cos[pos, :]), ol.load(m.sin[pos, :])
    ic, is_ = ol.load(m.one_c[0, :]), ol.load(m.one_s[0, :])
    for it in ol.range(N.DEC_LAYERS):
        s, cr = m.dec_layer(it)
        x.set(_attention(x, s, c, s_, pos, SPEC, block))
        x.set(_attention(x, cr, ic, is_, pos, SPEC, block, seq=T, cross=True))
    xs = ol.quantize(rmsnorm(x, ol.load(m.dec_norm), N.EPS))
    ol.store(m.logits, ol.dot(xs, m.head))


class Engine:
    """Needle on the ISA simulator (or a backend with run / read): encode(ids) once, then
    step(tok) per decoder token -> logits [V]."""

    def __init__(self, W: dict, cfg, enc_cap: int = 128, dec_cap: int = 128, backend=None,
                 cq: dict | None = None):
        self.img = Image(cfg, W, enc_cap, dec_cap, cq=cq)
        self.cfg = cfg
        self.backend = backend or Q3.IsaBackend(cfg, [self.img.build()], adopt=True)
        self.T, self.pos = 0, 0

    def _run(self, kernel, **kw):
        prog = trace_fit(kernel, self.cfg, 0, lambda: {"m": self.img, **kw}).finish()
        self.backend.run([prog])

    def encode(self, ids):
        if len(ids) > self.img.enc_cap:
            raise ValueError("prompt longer than the encoder capacity")
        self._run(needle_encode, ids=tuple(int(t) for t in ids))
        self.T, self.pos = len(ids), 0

    def step(self, tok: int) -> np.ndarray:
        if self.pos >= self.img.dec_cap:
            raise RuntimeError("decoder KV full")
        self._run(needle_step, pos=self.pos, tok=int(tok), T=self.T)
        self.pos += 1
        a = self.img.logits.base.const
        raw = self.backend.read(0, int(a), 4 * self.img.V)
        return raw.view(np.float32).copy()

    def generate(self, ids, n: int = 64) -> list:
        self.encode(ids)
        out, t = [], N.EOS
        for tk in (N.EOS, N.TOOL_CALL):
            lg = self.step(tk)
        while len(out) < n:
            t = int(lg.argmax())
            out.append(t)
            if t == N.EOS:
                break
            lg = self.step(t)
        return out
