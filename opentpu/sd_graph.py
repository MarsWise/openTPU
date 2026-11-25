"""Stable Diffusion 1.5 on the device as resident graphs (opentpu/resident.py): the CLIP text
encoder, the UNet (UNet2DConditionModel: ResnetBlock2D, Transformer2DModel with self- and
cross-attention and GEGLU, down / up sampling, skip concatenations), DDIM steps with
classifier-free guidance, and the VAE decoder -- every model operation a device program; the
initial latent drawn on the device (VOP RAND, Box-Muller).

The host's part: the prompt's tokens (CLIP's BPE), the schedule (timesteps and DDIM's
coefficients: constants, as RoPE tables are), the seed; it reads the image back.

Layout: images channels-last [H, W, C] (resident.T2); a convolution's input is a bordered copy
(GroupNorm writes one directly), token tensors are the same buffers seen as [H*W, C] rows
(resident.DT). Weights int8 MX on the MXU (Graph.weight / conv2d's packing), norms and the
scheduler in fp32 on the VPU. Names follow diffusers' state dicts (unet.*, vae.*, text.*)."""
from __future__ import annotations

import math

import numpy as np

from . import language as ol
from . import resident as RG
from .compiler import Tensor
from .kernels import randn as RN


# ================================================================ kernels
@ol.jit
def geglu_k(h, g, y):
    """y = h * gelu(g) (diffusers GEGLU: the projection's halves, the exact erf GELU)."""
    from .kernels import lib
    a = ol.load(h)
    a.set(a * lib.gelu(ol.load(g)))
    ol.store(y, a)


@ol.jit
def cfg_ddim_k(x, eu, ec, y, gs: float, a: float, b: float):
    """y = a x + b (eu + gs (ec - eu)): classifier-free guidance and one DDIM step (eta 0)
    folded into its two coefficients (the host's schedule constants)."""
    u = ol.load(eu)
    e = u + (ol.load(ec) - u) * gs
    ol.store(y, ol.load(x) * a + e * b)


class SDGraph(RG.Graph):
    """A Graph whose constants (weights, vectors) live in their own region from `cbase` (the
    activations bump from 0) and are shared: the same array's packing is placed once (keyed
    by its bytes), so the UNet's 2 x steps evaluations use one copy of its weights, and the
    activation arena can be reused from one evaluation to the next (`arena` / `reset`)."""

    def __init__(self, cfg, cbase: int):
        super().__init__(cfg)
        self.cbase = self.ctop = cbase
        self._memo: dict = {}

    def alloc(self, nbytes: int) -> int:
        a = super().alloc(nbytes)
        if self.top > self.cbase:
            raise MemoryError(f"activations reach {self.top:#x}, past the constants' {self.cbase:#x}")
        return a

    def const(self, a):
        a = np.ascontiguousarray(a)
        at = -(-self.ctop // RG.ALIGN) * RG.ALIGN
        self.ctop = at + a.nbytes
        self.consts.append((at, a))
        return at

    def _key(self, kind, a):
        import hashlib
        a = np.ascontiguousarray(a, np.float32)
        return (kind, a.shape, hashlib.blake2b(a.tobytes(), digest_size=16).digest())

    def weight(self, w):
        k = self._key("w", w)
        if k not in self._memo:
            self._memo[k] = super().weight(w)
        return self._memo[k]

    def vec(self, v):
        k = self._key("v", v)
        if k not in self._memo:
            self._memo[k] = super().vec(v)
        return self._memo[k]

    def conv2d(self, x, w, b, stride: int, pad: int, name="conv2d"):
        """resident's conv2d with the packed weights shared (memo by their bytes)."""
        from .compiler import QTensor
        from .kernels import conv as C
        from . import quant as Q
        Cout, Cin, KH, KW = w.shape
        OH, OW, Hp, Wp, Kp, rw = C.geometry(x.H, x.W, Cin, KH, KW, stride, pad, self.D)
        y = self.tensor2(OH, OW, Cout)
        k = self._key("c", w)
        if k not in self._memo:
            p = C.pack_conv_weight(np.asarray(w, np.float32), self.D)
            q, sc = Q.quantize_mxu(p, "int8", self.D)
            self._memo[k] = QTensor.weight(self.const(q), self.const(sc), p.shape[0], p.shape[1],
                                           "int8", self.D)
        b = np.zeros(Cout, np.float32) if b is None else b
        self.op(name, C.conv2d, x=x.rows(pad, rw), w=self._memo[k], b=self.vec(b),
                y=y.interior(), H=x.H, W=x.W, Cin=Cin, Cout=Cout, KH=KH, KW=KW, stride=stride,
                pad=pad, act="none")
        return y

    def image(self, size=None):
        top = max(self.top, self.ctop)
        return super().image(size or -(-top // RG.ALIGN) * RG.ALIGN)


ZW = 4096                                   # words a zero-fill row (an [8, ZW] tile)


@ol.jit
def zero_k(y, n: int):
    """y [n, ZW] = 0: an arena's stale words cleared before its next evaluation's tensors (their
    borders, pads and tails are read as zeros)."""
    t = ol.zeros([8, ZW])
    for r in range(0, n, 8):
        m = min(8, n - r)
        ol.store(y[r:r + m, :], t[0:m, :])


def zero_region(g, lo: int, hi: int, at: int, name: str) -> None:
    """Insert, at op index `at`, programs zeroing DRAM [lo, hi) (whole ZW-word rows)."""
    rows = -(-(hi - lo) // (4 * ZW))
    ops = []
    for i, r0 in enumerate(range(0, rows, 256)):
        n = min(256, rows - r0)
        ops.append((f"{name}.{i}", zero_k,
                    dict(y=Tensor(lo + 4 * ZW * r0, (n, ZW), (ZW, 1)), n=n)))
    g.ops[at:at] = ops


def _dt(t: RG.T2) -> RG.DT:
    """A dense image [H, W, C] as token rows [H*W, C]."""
    if t.border or t.tail:
        raise ValueError("a bordered image is not a token tensor")
    return RG.DT(t.addr, t.H * t.W, t.C)


def _t2(d: RG.DT, H: int, W: int) -> RG.T2:
    if d.pad or d.rs != d.C or d.L != H * W:
        raise ValueError("token rows are not a dense image")
    return RG.T2(d.addr, H, W, d.C)


class Weights:
    """SD 1.5's diffusers safetensors (fp16 or fp32) read lazily as fp32 under prefixes:
    text. (text_encoder), unet. (one file or shards), vae. -- keys -> files."""

    def __init__(self, text=None, unet=(), vae=None):
        from safetensors import safe_open
        self.files, self.index = [], {}
        for pre, paths in (("text.", [text] if text else []), ("unet.", list(unet)),
                           ("vae.", [vae] if vae else [])):
            for p in paths:
                f = safe_open(str(p), "np")
                self.files.append(f)
                for k in f.keys():
                    self.index[pre + k] = (f, k)

    def __contains__(self, k):
        return k in self.index

    def __getitem__(self, k):
        f, key = self.index[k]
        return np.asarray(f.get_tensor(key), np.float32)

    def keys(self):
        return self.index.keys()


class SD:
    """SD 1.5's weights W (diffusers names, prefixed text. / unet. / vae.; float arrays or
    lazily read), building onto one Graph."""

    def __init__(self, W, heads: int = 8):
        self.W, self.heads = W, heads

    def w(self, k):
        return np.asarray(self.W[k], np.float32)

    def has(self, k):
        return k in self.W

    # ---- helpers
    def bordered(self, g, x: RG.T2, name) -> RG.T2:
        """A dense image copied into a 1-pixel zero border with a D-word row tail (a 3x3
        convolution's input): one row move per image row."""
        y = g.tensor2(x.H, x.W, x.C, 1, g.D)
        src = Tensor(x.addr, (x.H, x.W * x.C), (x.W * x.C, 1))
        dst = Tensor(y.addr, (x.H + 2, y.rs), (y.rs, 1))
        g.copy_rows(src, dst, [(r, r + 1, x.C) for r in range(x.H)], x.W * x.C, name=name)
        return y

    def conv(self, g, x: RG.T2, p, stride=1, name="conv") -> RG.T2:
        w = self.w(p + "weight")
        k = w.shape[2]
        if k == 1:
            y = g.linear(_dt(x), w[:, :, 0, 0], self.w(p + "bias") if self.has(p + "bias") else None,
                         name=name)
            return _t2(y, x.H, x.W)
        if x.border < 1:
            x = self.bordered(g, x, name + ".border")
        return g.conv2d(x, w, self.w(p + "bias"), stride, 1, name=name)

    def gn(self, g, x: RG.T2, p, eps, act="none", border=1, name="gn") -> RG.T2:
        return g.group_norm2d(x, 32, self.w(p + "weight"), self.w(p + "bias"), eps, act=act,
                              border=border, tail=g.D if border else 0, name=name)

    def add(self, g, a: RG.T2, b: RG.T2, name) -> RG.T2:
        return _t2(g.add(_dt(a), _dt(b), name=name), a.H, a.W)

    def upsample2(self, g, x: RG.T2, name) -> RG.T2:
        """Nearest x 2 into a bordered buffer (the next 3x3 convolution's input)."""
        H, W, C = x.H, x.W, x.C
        y = g.tensor2(2 * H, 2 * W, C, 1, g.D)
        src = Tensor(x.addr, (H * W, C), (C, 1))
        dst = Tensor(y.addr, (2 * H + 2, y.rs), (y.rs, 1))
        moves = [(r * W + c, 2 * r + dy + 1, (2 * c + dx + 1) * C)
                 for r in range(H) for c in range(W) for dy in (0, 1) for dx in (0, 1)]
        g.copy_rows(src, dst, moves, C, name=name)
        return y

    def concat(self, g, a: RG.T2, b: RG.T2, name) -> RG.T2:
        y = g.tensor(a.H * a.W, a.C + b.C)
        g.copy(_dt(a), y.cols(0, a.C), name=name + ".a")
        g.copy(_dt(b), y.cols(a.C, a.C + b.C), name=name + ".b")
        return _t2(y, a.H, a.W)

    def attention(self, g, x: RG.DT, ctx: RG.DT | None, p, heads, name) -> RG.DT:
        C = x.C
        kv = x if ctx is None else ctx
        bq = self.w(p + "to_q.bias") if self.has(p + "to_q.bias") else None
        bk = self.w(p + "to_k.bias") if self.has(p + "to_k.bias") else None
        bv = self.w(p + "to_v.bias") if self.has(p + "to_v.bias") else None
        q = g.linear(x, self.w(p + "to_q.weight"), bq, name=name + ".q")
        k = g.linear(kv, self.w(p + "to_k.weight"), bk, name=name + ".k")
        v = g.linear(kv, self.w(p + "to_v.weight"), bv, name=name + ".v")
        o = g.attention(q, k, v, heads, heads, C // heads, [(0, x.L, kv.L)], name=name + ".attn")
        return g.linear(o, self.w(p + "to_out.0.weight"), self.w(p + "to_out.0.bias"),
                        name=name + ".out")

    # ---- CLIP text encoder
    def clip(self, g, ids) -> RG.DT:
        P = "text.text_model."
        T, H, heads = len(ids), 768, 12
        if getattr(self, "_tok", None) is None:            # (one table for every prompt)
            tok = self.w(P + "embeddings.token_embedding.weight")
            self._tok = RG.DT(g.const(tok), tok.shape[0], H)
            pe = self.w(P + "embeddings.position_embedding.weight")
            self._pos = RG.DT(g.const(pe), pe.shape[0], H)
        tab = self._tok
        x = g.tensor(T, H)
        g.copy_rows(tab.rows(), x.rows(), [(int(i), t, 0) for t, i in enumerate(ids)], H,
                    name="clip.tok")
        x = g.add(x, RG.DT(self._pos.addr, T, H), name="clip.pos")
        causal = [(r, r + 1, r + 1) for r in range(T)]
        for i in range(12):
            L = f"{P}encoder.layers.{i}."
            n = f"clip.l{i}"
            h = g.layer_norm(x, self.w(L + "layer_norm1.weight"), self.w(L + "layer_norm1.bias"),
                             1e-5, name=n + ".ln1")
            q = g.linear(h, self.w(L + "self_attn.q_proj.weight"), self.w(L + "self_attn.q_proj.bias"),
                         name=n + ".q")
            k = g.linear(h, self.w(L + "self_attn.k_proj.weight"), self.w(L + "self_attn.k_proj.bias"),
                         name=n + ".k")
            v = g.linear(h, self.w(L + "self_attn.v_proj.weight"), self.w(L + "self_attn.v_proj.bias"),
                         name=n + ".v")
            o = g.attention(q, k, v, heads, heads, H // heads, causal, name=n + ".attn")
            o = g.linear(o, self.w(L + "self_attn.out_proj.weight"),
                         self.w(L + "self_attn.out_proj.bias"), name=n + ".o")
            x = g.add(x, o, name=n + ".res1")
            h = g.layer_norm(x, self.w(L + "layer_norm2.weight"), self.w(L + "layer_norm2.bias"),
                             1e-5, name=n + ".ln2")
            h = g.linear(h, self.w(L + "mlp.fc1.weight"), self.w(L + "mlp.fc1.bias"),
                         act="quick_gelu", name=n + ".fc1")
            h = g.linear(h, self.w(L + "mlp.fc2.weight"), self.w(L + "mlp.fc2.bias"), name=n + ".fc2")
            x = g.add(x, h, name=n + ".res2")
        return g.layer_norm(x, self.w(P + "final_layer_norm.weight"),
                            self.w(P + "final_layer_norm.bias"), 1e-5, name="clip.ln_f")

    # ---- UNet pieces
    def resnet(self, g, x: RG.T2, temb: RG.DT | None, p, eps, name) -> RG.T2:
        h = self.gn(g, x, p + "norm1.", eps, "silu", name=name + ".gn1")
        h = self.conv(g, h, p + "conv1.", name=name + ".conv1")
        if temb is not None:
            t = g.linear(temb, self.w(p + "time_emb_proj.weight"), self.w(p + "time_emb_proj.bias"),
                         name=name + ".temb")
            g.add_row(_dt(h), Tensor(t.addr + 4 * t.pad * t.rs, (t.C,), (1,)),
                      name=name + ".add_temb")
        h = self.gn(g, h, p + "norm2.", eps, "silu", name=name + ".gn2")
        h = self.conv(g, h, p + "conv2.", name=name + ".conv2")
        sc = x
        if x.border:
            raise ValueError("resnet input must be dense")
        if self.has(p + "conv_shortcut.weight"):
            sc = self.conv(g, x, p + "conv_shortcut.", name=name + ".short")
        return self.add(g, h, sc, name + ".res")

    def transformer(self, g, x: RG.T2, ctx: RG.DT, p, name) -> RG.T2:
        C = x.C
        heads = self.heads
        h = self.gn(g, x, p + "norm.", 1e-6, border=0, name=name + ".gn")
        h = self.conv(g, h, p + "proj_in.", name=name + ".proj_in")
        t = _dt(h)
        B = p + "transformer_blocks.0."
        n1 = g.layer_norm(t, self.w(B + "norm1.weight"), self.w(B + "norm1.bias"), 1e-5,
                          name=name + ".ln1")
        t = g.add(t, self.attention(g, n1, None, B + "attn1.", heads, name + ".self"),
                  name=name + ".r1")
        n2 = g.layer_norm(t, self.w(B + "norm2.weight"), self.w(B + "norm2.bias"), 1e-5,
                          name=name + ".ln2")
        t = g.add(t, self.attention(g, n2, ctx, B + "attn2.", heads, name + ".cross"),
                  name=name + ".r2")
        n3 = g.layer_norm(t, self.w(B + "norm3.weight"), self.w(B + "norm3.bias"), 1e-5,
                          name=name + ".ln3")
        pw, pb = self.w(B + "ff.net.0.proj.weight"), self.w(B + "ff.net.0.proj.bias")
        F = pw.shape[0] // 2
        hh = g.linear(n3, pw[:F], pb[:F], name=name + ".ff_h")
        gg = g.linear(n3, pw[F:], pb[F:], name=name + ".ff_g")
        y = g.tensor(t.L, F)
        g._rowwise(name + ".geglu", geglu_k, t.L, F,
                   lambda r0, r1: dict(h=hh.rows(r0, r1), g=gg.rows(r0, r1), y=y.rows(r0, r1)),
                   tiles=8)                     # h, g and the erf GELU's temporaries
        f = g.linear(y, self.w(B + "ff.net.2.weight"), self.w(B + "ff.net.2.bias"),
                     name=name + ".ff_out")
        t = g.add(t, f, name=name + ".r3")
        o = self.conv(g, _t2(t, x.H, x.W), p + "proj_out.", name=name + ".proj_out")
        return self.add(g, o, x, name + ".res")

    def time_embedding(self, g, t_in: RG.DT) -> RG.DT:
        P = "unet.time_embedding."
        h = g.linear(t_in, self.w(P + "linear_1.weight"), self.w(P + "linear_1.bias"),
                     act="silu", name="unet.temb1")
        h = g.linear(h, self.w(P + "linear_2.weight"), self.w(P + "linear_2.bias"), name="unet.temb2")
        return g.unary(h, "silu", name="unet.temb_act")     # (every resnet's SiLU(temb))

    def unet(self, g, x: RG.T2, t_in: RG.DT, ctx: RG.DT, tag="u") -> RG.T2:
        """One UNet evaluation: x [h, w, 4] dense latent, t_in [1, 320] the sinusoid row,
        ctx [77, 768] the text states -> eps [h, w, 4]."""
        U = "unet."
        eps = 1e-5
        temb = self.time_embedding(g, t_in)
        h = self.conv(g, x, U + "conv_in.", name=f"{tag}.conv_in")
        skips = [h]
        for i in range(4):
            D = f"{U}down_blocks.{i}."
            for j in range(2):
                h = self.resnet(g, h, temb, f"{D}resnets.{j}.", eps, f"{tag}.d{i}r{j}")
                if self.has(f"{D}attentions.{j}.norm.weight"):
                    h = self.transformer(g, h, ctx, f"{D}attentions.{j}.", f"{tag}.d{i}a{j}")
                skips.append(h)
            if i < 3:
                h = self.conv(g, h, f"{D}downsamplers.0.conv.", stride=2, name=f"{tag}.d{i}down")
                skips.append(h)
        M = U + "mid_block."
        h = self.resnet(g, h, temb, M + "resnets.0.", eps, f"{tag}.m.r0")
        h = self.transformer(g, h, ctx, M + "attentions.0.", f"{tag}.m.a0")
        h = self.resnet(g, h, temb, M + "resnets.1.", eps, f"{tag}.m.r1")
        for i in range(4):
            Up = f"{U}up_blocks.{i}."
            for j in range(3):
                h = self.concat(g, h, skips.pop(), f"{tag}.u{i}c{j}")
                h = self.resnet(g, h, temb, f"{Up}resnets.{j}.", eps, f"{tag}.u{i}r{j}")
                if self.has(f"{Up}attentions.{j}.norm.weight"):
                    h = self.transformer(g, h, ctx, f"{Up}attentions.{j}.", f"{tag}.u{i}a{j}")
            if i < 3:
                h = self.conv(g, self.upsample2(g, h, f"{tag}.u{i}up"),
                              f"{Up}upsamplers.0.conv.", name=f"{tag}.u{i}upconv")
        h = self.gn(g, h, U + "conv_norm_out.", eps, "silu", name=f"{tag}.gn_out")
        return self.conv(g, h, U + "conv_out.", name=f"{tag}.conv_out")

    # ---- the VAE decoder
    def vae_decode(self, g, z: RG.T2) -> RG.T2:
        V = "vae."
        eps = 1e-6
        zs = _t2(g.scale(_dt(z), 1 / 0.18215, y=g.tensor(z.H * z.W, z.C), name="vae.scale"),
                 z.H, z.W)
        h = self.conv(g, zs, V + "post_quant_conv.", name="vae.pq")
        h = self.conv(g, h, V + "decoder.conv_in.", name="vae.conv_in")
        M = V + "decoder.mid_block."
        h = self.resnet(g, h, None, M + "resnets.0.", eps, "vae.m.r0")
        A = M + "attentions.0."
        a = self.gn(g, h, A + "group_norm.", eps, border=0, name="vae.m.attn_gn")
        o = self.attention(g, _dt(a), None, A, 1, "vae.m.attn")
        h = self.add(g, _t2(o, h.H, h.W), h, "vae.m.attn_res")
        h = self.resnet(g, h, None, M + "resnets.1.", eps, "vae.m.r1")
        for i in range(4):
            Up = f"{V}decoder.up_blocks.{i}."
            for j in range(3):
                h = self.resnet(g, h, None, f"{Up}resnets.{j}.", eps, f"vae.u{i}r{j}")
            if i < 3:
                h = self.conv(g, self.upsample2(g, h, f"vae.u{i}up"), f"{Up}upsamplers.0.conv.",
                              name=f"vae.u{i}upconv")
        h = self.gn(g, h, V + "decoder.conv_norm_out.", eps, "silu", name="vae.gn_out")
        return self.conv(g, h, V + "decoder.conv_out.", name="vae.conv_out")


# ================================================================ the schedule (host constants)
def ddim_schedule(steps: int, train: int = 1000, beta_start=0.00085, beta_end=0.012,
                  offset: int = 1):
    """DDIMScheduler (scaled_linear betas, set_alpha_to_one False, steps_offset 1, "leading"
    spacing): the timesteps and each step's (a, b) with x_prev = a x + b eps (eta 0)."""
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, train, dtype=np.float64) ** 2
    ac = np.cumprod(1 - betas)
    ratio = train // steps
    ts = (np.arange(0, steps) * ratio).round()[::-1].astype(np.int64) + offset
    out = []
    for t in ts:
        prev = t - ratio
        at = ac[t]
        ap = ac[prev] if prev >= 0 else ac[0]
        a = math.sqrt(ap / at)
        b = math.sqrt(1 - ap) - math.sqrt(ap) * math.sqrt(1 - at) / math.sqrt(at)
        out.append((int(t), a, b))
    return out


def timestep_row(t: int, dim: int = 320) -> np.ndarray:
    """diffusers Timesteps(320, flip_sin_to_cos=True, downscale_freq_shift=0): [cos | sin]."""
    half = dim // 2                              # (in fp32, as diffusers computes it)
    f = np.exp(np.float32(-math.log(10000)) * np.arange(half, dtype=np.float32)
               / np.float32(half)).astype(np.float32)
    e = (np.float32(t) * f).astype(np.float32)
    return np.concatenate([np.cos(e), np.sin(e)]).astype(np.float32)


def _layout(ops, lo: int = 0) -> tuple:
    """Ops' kernels and the addresses / shapes / strides of their views at or above `lo` (the
    arena), names left out; a view below it (an input, a constant, a kept output) by its shape
    alone. Two evaluations with equal layouts read and write the same arena words."""
    from .compiler import Affine

    def arg(v):
        if hasattr(v, "base") and hasattr(v, "shape"):
            try:
                a = Affine.of(v.base).static() + 4 * Affine.of(v.offset).static()
            except Exception:               # (a loop-dependent address: kept whole)
                a = None
            shp = (tuple(v.shape), tuple(getattr(v, "strides", ())))
            if a is not None and a < lo:
                return ("ext",) + shp
            return ("T", a if a is not None else repr(v.base)) + shp
        return v if isinstance(v, (int, float, str, bool, type(None))) else repr(v)
    return tuple((id(k), tuple(sorted((n, arg(v)) for n, v in a.items()))) for _, k, a in ops)


def build_txt2img(g, sd: SD, ids_c, ids_u, h: int, w: int, steps: int, gs: float = 7.5):
    """The whole pipeline on one Graph: both prompts' CLIP states, the initial latent from the
    seed (VOP RAND), `steps` DDIM steps of two UNet evaluations each, the VAE decoder.
    -> (image T2 [8h, 8w, 3] in [-1, 1] before clamping, the seed tensor, the latent)."""
    ctx_c = sd.clip(g, ids_c)
    ctx_u = sd.clip(g, ids_u)
    n = h * w * 4
    x = g.tensor(h * w, 4)
    from . import rand as RD
    cnt = g.const(RD.bits(RD.counters(n)).view(np.float32))
    seed = g.tensor(1, 1)
    g.op("sd.noise", RN.seeded_normal_rows, c=Tensor(cnt, (n,), (1,)),
         seed=Tensor(seed.addr, (1,), (1,)), y=Tensor(x.addr, (1, n), (n, 1)), M=1, N=n, key=1)
    sched = ddim_schedule(steps)
    lat = [x, g.tensor(h * w, 4)]                       # ping-pong latents
    ec_keep, eu_keep = g.tensor(h * w, 4), g.tensor(h * w, 4)
    trs = []
    for s, (t, a, b) in enumerate(sched):
        tr = g.tensor(1, 320)
        trs.append(tr)
    arena = -(-g.top // RG.ALIGN) * RG.ALIGN
    used, last = [arena], None
    for s, (t, a, b) in enumerate(sched):
        g.consts.append((trs[s].addr, timestep_row(t)[None, :]))
        xin, xout = lat[s % 2], lat[(s + 1) % 2]
        for ctx, keep, tag in ((ctx_c, ec_keep, "c"), (ctx_u, eu_keep, "u")):
            g.top = arena                               # every evaluation's activations reuse it
            at = len(g.ops)
            e = sd.unet(g, _t2(xin, h, w), trs[s], ctx, tag=f"s{s}{tag}")
            g.copy(_dt(e), keep, name=f"s{s}{tag}.keep")
            sig = _layout(g.ops[at:], arena)
            # (stale: the previous evaluation's). An evaluation with the previous one's layout
            # (every op the same kernel on the same addresses: the UNet's 2 x steps) writes
            # what it reads first, as that one did, and never a border: no zeroing
            if max(used) > arena and sig != last:
                zero_region(g, arena, max(used), at, f"s{s}{tag}.zero")
            last = sig
            used.append(g.top)
        g.op(f"s{s}.ddim", cfg_ddim_k, x=xin.rows(), eu=eu_keep.rows(), ec=ec_keep.rows(),
             y=xout.rows(), gs=float(gs), a=float(a), b=float(b))
    x = lat[len(sched) % 2]
    g.top = arena
    at = len(g.ops)
    img = sd.vae_decode(g, _t2(x, h, w))
    zero_region(g, arena, max(used), at, "vae.zero")
    return img, seed, x


def compile_cached(g, ops, cache: dict) -> list:
    """g.compile per op, with identical ops (the same kernel and arguments: the UNet's
    evaluations step after step reuse one arena) compiled once: [(name, program)]."""
    out = []
    for name, kernel, args in ops:
        key = (id(kernel), repr(sorted(args.items(), key=lambda kv: kv[0])))
        if key not in cache:
            cache[key] = g.compile([(name, kernel, args)])[0][1]
        out.append((name, cache[key]))
    return out


__all__ = ["SD", "compile_cached", "ddim_schedule", "timestep_row", "build_txt2img", "geglu_k", "cfg_ddim_k"]
