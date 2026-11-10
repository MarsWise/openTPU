"""PaddleOCR-VL 1.6 on the device: the vision tower and projector as kernels/vit.py phases (one
program each, rows in hardware-loop groups) and the ERNIE decoder on qwen3's engine.

Vision (VisionImage): the 27 layers' weights int8 (the MLP's 4304 padded to 4352: zero rows /
columns), each LayerNorm's bias folded into the next linear's (vit_qkv's layernorm_q has
none: W (y + b) = W y + W b); the patches in window order (each 2 x 2 window's 4 patches
consecutive, the windows raster over the (h / 2, w / 2) grid), their position-table rows
(the 27 x 27 table bilinearly resized to the grid: a per-resolution constant) and 2-D RoPE
rows in that order (attention does not see the order; the merge then reads each window's 4
rows as one [4608] row, the projector's rows raster: the decoder's image tokens).
Phases: vit_embed; per layer vit_qkv (with RoPE) / vit_attend / vit_out_mlp (the last with
post_layernorm); ln_rows (the projector's pre_norm); proj (linear_1, GELU erf, linear_2).

The decoder: qwen3.Engine (int8), the rows' embeddings / mRoPE cos-sin rows written by the
host (the image rows read from the vision image's DRAM: data movement; their on-card hand-off
into the decoder's prefill is the MiniCPM-V path's, not used here yet).
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from .. import fp32 as F
from .. import language as ol
from .. import quant as Q
from ..compiler import Affine, QTensor, Tensor, trace_fit
from ..kernels import vit as VK
from ..kernels.lib import gelu
from . import paddleocr_vl as P
from .qwen3 import _tdesc

C, HEADS, DV, LAYERS, FFN = 1152, 16, 72, 27, 4304
CHUNK = 256
FP = -(-FFN // CHUNK) * CHUNK          # 4352
KP = 640                                # 3 * 14 * 14 = 588 padded to D multiples
V = "visual.vision_model."


def window_order(gh: int, gw: int) -> np.ndarray:
    """Raster patch index of each row in window order."""
    out = []
    for i in range(gh // 2):
        for j in range(gw // 2):
            for di in range(2):
                for dj in range(2):
                    out.append((2 * i + di) * gw + 2 * j + dj)
    return np.asarray(out)


class _Bump:
    def __init__(self, base=4096):
        self.next = base

    def alloc(self, n, align=128):
        a = -(-self.next // align) * align
        self.next = a + n
        return a


class VisionImage:
    def __init__(self, cfg, W, nmax: int = 1024, fmt: str = "int8"):
        D = cfg.D
        self.cfg, self.D, self.fmt, self.nmax = cfg, D, fmt, -(-nmax // D) * D
        self.dp = -(-DV // D) * D
        b = _Bump()
        self.puts = []
        f32 = lambda a: F.ftz(np.asarray(a, np.float32))            # noqa: E731
        g = lambda k: W[k].numpy().astype(np.float64)                 # noqa: E731

        def vec(a):
            a = f32(a)
            addr = b.alloc(a.nbytes)
            self.puts.append((addr, a))
            return _tdesc(addr, (a.size,))

        def mat(w):
            w = np.asarray(w, np.float32)
            q, sc = Q.quantize_mxu(w, fmt, D)
            a0, a1 = b.alloc(q.nbytes), b.alloc(sc.nbytes)
            self.puts += [(a0, q), (a1, sc)]
            return QTensor.weight(a0, a1, w.shape[0], w.shape[1], fmt, D)

        self.layers = []
        for i in range(LAYERS):
            p = f"{V}encoder.layers.{i}."
            l1b, l2b = g(p + "layer_norm1.bias"), g(p + "layer_norm2.bias")
            wq, wk, wv = (g(p + f"self_attn.{n}_proj.weight") for n in "qkv")
            w1 = np.zeros((FP, C))
            w1[:FFN] = g(p + "mlp.fc1.weight")
            b1 = np.zeros(FP)
            b1[:FFN] = g(p + "mlp.fc1.bias") + g(p + "mlp.fc1.weight") @ l2b
            w2 = np.zeros((C, FP))
            w2[:, :FFN] = g(p + "mlp.fc2.weight")
            self.layers.append(SimpleNamespace(
                ln1w=vec(g(p + "layer_norm1.weight")), ln2w=vec(g(p + "layer_norm2.weight")),
                bq=vec(g(p + "self_attn.q_proj.bias") + wq @ l1b),
                bk=vec(g(p + "self_attn.k_proj.bias") + wk @ l1b),
                bv=vec(g(p + "self_attn.v_proj.bias") + wv @ l1b),
                bo=vec(g(p + "self_attn.out_proj.bias")), b1=vec(b1),
                b2=vec(g(p + "mlp.fc2.bias")), wq=mat(wq), wk=mat(wk), wv=mat(wv),
                wo=mat(g(p + "self_attn.out_proj.weight")), w1=mat(w1), w2=mat(w2)))
        wp = np.zeros((C, KP))
        wp[:, :588] = g(V + "embeddings.patch_embedding.weight").reshape(C, -1)
        self.emb = SimpleNamespace(wp=mat(wp), bp=vec(g(V + "embeddings.patch_embedding.bias")))
        self.post = (vec(g(V + "post_layernorm.weight")), vec(g(V + "post_layernorm.bias")))
        self.pre = (vec(g("mlp_AR.pre_norm.weight")), vec(g("mlp_AR.pre_norm.bias")))
        self.proj = SimpleNamespace(w1=mat(g("mlp_AR.linear_1.weight")),
                                    b1=vec(g("mlp_AR.linear_1.bias")),
                                    w2=mat(g("mlp_AR.linear_2.weight")),
                                    b2=vec(g("mlp_AR.linear_2.bias")))
        self.pos_table = g(V + "embeddings.position_embedding.weight")
        n = self.nmax
        w = {k: b.alloc(sz) for k, sz in dict(P=4 * n * KP, E=4 * n * C, X=4 * n * C,
                                               Q=4 * n * HEADS * self.dp, O=4 * n * C,
                                               RC=4 * n * DV // 2, RS=4 * n * DV // 2,
                                               vis=4 * (n // 4) * 1024).items()}
        self.kvh = {h: {"k": b.alloc(n * self.dp), "ks": b.alloc(4 * n),
                        "vt": b.alloc(self.dp * n), "vs": b.alloc(4 * n)} for h in range(HEADS)}
        self.work = w
        self.nbytes = -(-b.next // 4096) * 4096
        if self.nbytes > cfg.DRAM_BYTES:
            raise MemoryError("vision image > DRAM_BYTES")

    def build(self) -> np.ndarray:
        img = np.zeros(self.nbytes, np.uint8)
        for addr, a in self.puts:
            v = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
            img[addr:addr + v.size] = v
        return img

    def inputs(self, patches: np.ndarray, gh: int, gw: int) -> list:
        """(address, array) writes of one image: its patch rows, position-table rows, RoPE rows,
        all in window order (pixels and per-resolution constants)."""
        import torch
        import torch.nn.functional as Fn
        N = gh * gw
        if N > self.nmax:
            raise ValueError(f"{N} patches > nmax {self.nmax}")
        o = window_order(gh, gw)
        p = np.zeros((N, KP), np.float32)
        p[:, :588] = patches[o]
        pe = torch.from_numpy(self.pos_table.astype(np.float32)).reshape(1, 27, 27, C)
        pe = Fn.interpolate(pe.permute(0, 3, 1, 2), size=(gh, gw), mode="bilinear",
                            align_corners=False).permute(0, 2, 3, 1).reshape(-1, C).numpy()
        rc, rs = P.vision_rope(gh, gw)
        w = self.work
        return [(w["P"], F.ftz(p)), (w["E"], F.ftz(pe[o].astype(np.float32))),
                (w["RC"], F.ftz(rc[o, :DV // 2])), (w["RS"], F.ftz(rs[o, :DV // 2]))]

    def v(self, N: int) -> SimpleNamespace:
        from ..kernels.vit import VitKV
        w, n = self.work, self.nmax
        return SimpleNamespace(
            C=C, heads=HEADS, dp=self.dp, dv=DV,
            P=_tdesc(w["P"], (n, KP)), E=_tdesc(w["E"], (n, C)), X=_tdesc(w["X"], (n, C)),
            Q=[_tdesc(w["Q"] + 4 * h * n * self.dp, (n, self.dp)) for h in range(HEADS)],
            O=_tdesc(w["O"], (n, C)), X4=_tdesc(w["X"], (n // 4, 4 * C)),
            vis=_tdesc(w["vis"], (n // 4, 1024)),
            rope=(_tdesc(w["RC"], (n, DV // 2)), _tdesc(w["RS"], (n, DV // 2))),
            kv=VitKV(self.kvh, n, self.dp, self.D, 1, 0, dv=DV), emb=self.emb)

    def programs(self, N: int, R: int = 16, block: int = 128) -> list:
        """The tower's and projector's programs, in run order."""
        cfg, v = self.cfg, self.v(N)
        tr = lambda k, **kw: trace_fit(k, cfg, 0, lambda: dict(v=v, **kw)).finish()  # noqa
        sc = 1.0 / math.log(2.0) / math.sqrt(DV)          # (exp2 softmax)
        out = [tr(VK.vit_embed, N=N, R=R)]
        for i, lw in enumerate(self.layers):
            out.append(tr(VK.vit_qkv, lw=lw, src=v.X, N=N, R=R, eps=1e-6))
            out.append(tr(VK.vit_attend, N=N, scale=sc, block=block))
            out.append(tr(VK.vit_out_mlp, lw=lw, dst=v.X, N=N, R=R, eps=1e-6, chunk=CHUNK,
                          post=self.post if i == LAYERS - 1 else None))
        out.append(tr(ln_rows, src=v.X, w=self.pre, N=N, R=R, eps=1e-5))
        out.append(tr(proj_rows, pw=self.proj, N4=N // 4, R=8))
        return out


@ol.jit
def ln_rows(v, src, w, N: int, R: int, eps: float):
    """src's N rows LayerNorm'd in place (gain, bias)."""
    g, bb = ol.load(w[0]), ol.load(w[1])

    def body(r0, n):
        x = ol.load(src[r0:r0 + n, :])
        VK.layernorm_(x, g, bb, eps)
        ol.store(src[r0:r0 + n, :], x)
    VK.row_groups(N, R, body)


@ol.jit
def proj_rows(v, pw, N4: int, R: int):
    """Each window's 4 rows as one [4C] row: linear_1, GELU (erf), linear_2 -> v.vis."""
    b2 = ol.load(pw.b2)

    def body(r0, n):
        xs = ol.quantize(ol.load(v.X4[r0:r0 + n, :]), free=True)
        y = VK.mlp_bias(xs, pw.w1, pw.b1, pw.w2, CHUNK, act=gelu, rows=n)
        y.set(y + b2[None, :])
        ol.store(v.vis[r0:r0 + n, :], y)
    VK.row_groups(N4, R, body)


def vision_device(cfg, W, patches, gh: int, gw: int, img=None):
    """Run the tower and projector on the ISA -> [N / 4, 1024] image rows (raster)."""
    from .qwen3 import IsaBackend
    img = img or VisionImage(cfg, W, nmax=gh * gw)
    be = IsaBackend(cfg, [img.build()], adopt=True)
    for a, arr in img.inputs(patches, gh, gw):
        be.write(0, a, arr)
    N = gh * gw
    for prog in img.programs(N):
        be.run([prog])
    raw = be.read(0, img.work["vis"], 4 * (N // 4) * 1024)
    return raw.view(np.float32).reshape(N // 4, 1024).copy(), be, img



def decoder_spec():
    from . import qwen3 as Q3
    return Q3.Spec(hidden=1024, layers=18, n_q=16, n_kv=2, head_dim=128, ffn=3072,
                   vocab=103424, eps=1e-5, theta=500000.0, tied=False, bos=1, eos=(2,),
                   qk_norm=False, qk_smooth=False)


class _NW:
    def __init__(self, W):
        self.W = W

    def __getitem__(self, k):
        return self.W[k].numpy()

    def __contains__(self, k):
        return k in self.W.h

    def keys(self):
        return self.W.keys()


def decode_device(cfg, W, tok, rows, gh: int, gw: int, task: str = "OCR:", n: int = 64,
                  cap: int = 512, R: int = 32) -> str:
    """The ERNIE decoder on the ISA (qwen3.Engine, int8): the prompt's rows (the image's from
    `rows`) with their mRoPE cos / sin rows, then greedy decode (positions shifted by the
    image's mRoPE delta) -> the text."""
    from . import qwen3 as Q3
    ids = P.prompt_ids(tok, gh, gw, task)
    pos = P.mrope_positions(ids, gh, gw)
    T = len(ids)
    if T + n > cap:
        raise ValueError("prompt + output above the KV capacity")
    cos3, sin3 = P.mrope_tables(pos)
    delta = int(pos.max()) + 1 - T
    cs = P.mrope_tables(np.tile(np.arange(cap)[None] + delta, (3, 1)))
    cosT, sinT = cs[0][:, :64].copy(), cs[1][:, :64].copy()
    cosT[:T], sinT[:T] = cos3[:, :64], sin3[:, :64]
    eng = Q3.Engine(decoder_spec(), _NW(W), cap=cap, cfg=cfg, resident=False, pipeline=False,
                    rows=R)
    emb, img = eng.embed, np.asarray(rows, np.float32)
    base = 1 << 30

    class _E:                      # token ids, the image's rows at keys base + j
        def __getitem__(self, ks):
            one = np.ndim(ks) == 0
            out = [img[k - base] if k >= base else np.asarray(emb[int(k)], np.float32)
                   for k in np.atleast_1d(ks)]
            return out[0] if one else np.stack(out)
    eng.embed = _E()
    eng._rope = [cosT, sinT]
    keys, j = [], 0
    for t in ids:
        keys.append(base + j if t == P.IMAGE_TOKEN else t)
        j += t == P.IMAGE_TOKEN
    lg = eng.prefill(keys)
    out = []
    for _ in range(n):
        t = int(np.argmax(lg))
        out.append(t)
        if t == 2:
            break
        lg = eng.step(t)
    return tok.decode([t for t in out if t != 2])
