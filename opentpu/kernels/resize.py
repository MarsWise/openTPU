"""Image resizes on the device: separable antialiased filters (Pillow's / torchvision's
precompute_coeffs: bilinear, or bicubic with a = -0.5), the vertical pass on the VPU and the
horizontal one on the MXU (kernels/letterbox.py's letterbox_rows), each output row's columns in
channel-major order ([c][x], what a channel-first patch layout reads contiguously) with an
affine folded into the horizontal weights and bias (a normalization: (x - m) / s).

MiniCPMVPrep: MiniCPM-V's processor on the device -- the thumbnail and the refined
resize of an image, both normalized ((x - 127.5) / 127.5), cut into the slices, every crop's
14 x 14 patches as channel-major rows [h w, 3 * 196] (padded to KP) in the tower's card order
(llm/minicpmv.py card_order): the host decodes the file and writes its RGB pixels only.
torchvision rounds its resize to uint8 first; this does not (about a level, 1 / 127.5 here).
"""
from __future__ import annotations

import numpy as np

from .. import language as ol
from ..compiler import QTensor, Tensor


def _cubic(x, a=-0.5):
    x = np.abs(x)
    return np.where(x < 1, ((a + 2) * x - (a + 3)) * x * x + 1,
                    np.where(x < 2, (((x - 5) * x + 8) * x - 4) * a, 0.0))


def coeffs(n_in: int, n_out: int, kind: str = "bilinear") -> list:
    """Per output pixel (first input pixel, normalized weights)."""
    support0, f = (1.0, lambda t: np.maximum(0.0, 1.0 - np.abs(t))) if kind == "bilinear" \
        else (2.0, _cubic)
    scale = n_in / n_out
    fs = max(scale, 1.0)
    support = fs * support0
    out = []
    for xx in range(n_out):
        center = (xx + 0.5) * scale
        lo = max(int(center - support + 0.5), 0)
        hi = min(int(center + support + 0.5), n_in)
        x = np.arange(lo, hi, dtype=np.float64)
        w = f((x - center + 0.5) / fs)
        if w.sum() == 0:
            w, x = np.ones(1), x[:1]
        out.append((lo, w / w.sum()))
    return out


class ResizePlan:
    """[H, W, 3] (HWC rows) -> [oh, 3 ow] rows, columns [c][x], times `scale` plus `bias`."""

    def __init__(self, H, W, oh, ow, kind="bicubic", scale=1.0, bias=0.0, clip=None):
        self.H, self.W, self.oh, self.ow = H, W, oh, ow
        self.clip = clip            # (lo, hi): the input's range, both passes clamped to it
        self.rows = coeffs(H, oh, kind)
        self.cols = coeffs(W, ow, kind)
        m = np.zeros((3 * ow, 3 * W), np.float32)
        for x, (lo, w) in enumerate(self.cols):
            for c in range(3):
                m[c * ow + x, 3 * (lo + np.arange(len(w))) + c] = w * scale
        self.hmat = m
        self.scale_ = scale
        self.bias = np.full(3 * ow, bias, np.float32)

    def reference(self, img) -> np.ndarray:
        x = np.asarray(img, np.float64).reshape(self.H, 3 * self.W)
        v = np.stack([w @ x[lo:lo + len(w)] for lo, w in self.rows])
        if self.clip:
            v = np.clip(v, *self.clip)
        y = v @ self.hmat.T.astype(np.float64) + self.bias
        return np.clip(y, *self.out_clip) if self.clip else y

    @property
    def out_clip(self) -> tuple:
        s, b = float(self.scale_), float(self.bias[0])
        return tuple(sorted((self.clip[0] * s + b, self.clip[1] * s + b)))

    def bands(self, D: int, max_blocks: int, tile: int = 128) -> tuple:
        out = []
        for c in range(3):
            x0 = 0
            while x0 < self.ow:
                x1 = min(self.ow, x0 + tile)
                while True:
                    lo = min(self.cols[j][0] for j in range(x0, x1))
                    hi = max(self.cols[j][0] + len(self.cols[j][1]) for j in range(x0, x1))
                    k0, k1 = (3 * lo) // D * D, -(-3 * hi // D) * D
                    if (k1 - k0) // D <= max_blocks or x1 - x0 == 1:
                        break
                    x1 = x0 + max(1, (x1 - x0) // 2)
                out.append((c * self.ow + x0, c * self.ow + x1, k0, k1))
                x0 = x1
        return tuple(out)

    def weights(self, D: int):
        from .. import quant as Q
        from .linear import pack_weight
        return Q.quantize_mxu(pack_weight(self.hmat, D), "int8", D)


def resize_programs(cfg, plan: ResizePlan, src: int, wdata: int, wscale: int, bias: int,
                    dst: int, rows: int = 8) -> list:
    """The resize as device programs: src [H, 3W] fp32 dense, dst [oh, 3 ow] dense."""
    from ..compiler import QACT_KB_MAX
    from ..runtime import compile_kernel
    from .letterbox import letterbox_rows
    D = cfg.D
    Kp = -(-3 * plan.W // D) * D
    N = 3 * plan.ow
    s_t = Tensor(src, (plan.H, 3 * plan.W), (3 * plan.W, 1))
    d_t = Tensor(dst, (plan.oh, N), (N, 1))
    w_t = QTensor.weight(wdata, wscale, N, Kp, "int8", D)
    b_t = Tensor(bias, (N,), (1,))
    bands = plan.bands(D, min(cfg.ACT_BLOCKS, QACT_KB_MAX) // 2)
    out = []
    for i0 in range(0, plan.oh, rows):
        i1 = min(plan.oh, i0 + rows)
        taps = tuple((lo, tuple(float(c) for c in w)) for lo, w in plan.rows[i0:i1])
        comp, _ = compile_kernel(letterbox_rows, cfg, src=s_t, w=w_t, bias=b_t, dst=d_t,
                                 i0=i0, i1=i1, taps=taps, pad_rows=False, bands=bands,
                                 vclip=tuple(plan.clip or ()),
                                 clip=plan.out_clip if plan.clip else ())
        out.append(comp.programs[0])
    return out


@ol.jit
def blocks_k(src, dst, moves: tuple):
    """2-D block moves in one program: (source word, destination word, rows, cols, source row
    stride, destination row stride) within the flat src / dst ranges."""
    for so, do, r, c, srs, drs in moves:
        t = ol.load(Tensor(src.base + 4 * so, (r, c), (srs, 1)))
        ol.store(Tensor(dst.base + 4 * do, (r, c), (drs, 1)), t)
        del t


def blocks_programs(cfg, src: int, dst: int, moves, per_prog: int = 1500) -> list:
    from ..runtime import compile_kernel
    s_t, d_t = Tensor(src, (1,), (1,)), Tensor(dst, (1,), (1,))
    out = []
    for i in range(0, len(moves), per_prog):
        comp, _ = compile_kernel(blocks_k, cfg, src=s_t, dst=d_t,
                                 moves=tuple(tuple(int(v) for v in m)
                                             for m in moves[i:i + per_prog]))
        out.append(comp.programs[0])
    return out


@ol.jit
def fill_k(dst, n: int, value: float):
    """dst's first n words (a [rows, cols] dense view of them) set to value."""
    cols = dst.shape[1]
    R = max(1, min(dst.shape[0], (ol.tmem_contiguous() - 1024) // cols))
    t = ol.full([R, cols], value)
    for r0 in range(0, dst.shape[0], R):
        m = min(R, dst.shape[0] - r0)
        ol.store(dst[r0:r0 + m, :], t[0:m, :])


def fill_programs(cfg, dst: int, rows: int, cols: int, value: float = 0.0) -> list:
    from ..runtime import compile_kernel
    comp, _ = compile_kernel(fill_k, cfg, dst=Tensor(dst, (rows, cols), (cols, 1)),
                             n=rows * cols, value=float(value))
    return [comp.programs[0]]


class MiniCPMVPrep:
    """MiniCPM-V's processor on the device for an [H, W, 3] image (fp32 pixel values 0..255,
    HWC, at `src`): the thumbnail's and the refined resize (bicubic, antialiased, normalized
    (x - 127.5) / 127.5) into scratch from `base` up, then per crop its patch rows [h w, KP]
    (zero past 3 patch^2) in the tower's card order into any `dst` (VisionImage.work["P"]).
    consts: [(address, array)] the host writes once per image size (the filters, as weights);
    programs(): the resizes; crop_programs(k, dst): crop k's rows. sizes: each crop's (h, w)
    in patches, as minicpmv.crops / prepare give them."""

    def __init__(self, cfg, H: int, W: int, vs, src: int, base: int, max_slices=None):
        from ..llm import minicpmv as MV
        D = cfg.D
        self.cfg, self.p = cfg, vs.patch
        p = vs.patch
        self.KP = KP = -(-3 * p * p // D) * D
        max_slices = vs.max_slices if max_slices is None else max_slices
        grid = MV.slice_grid(H, W, max_slices, vs.scale) if max_slices >= 1 else None
        th, tw = MV.best_resize(H, W, vs.scale, vs.patch, upscale=grid is None)
        resized = [(th, tw, [(0, 0, th, tw)])]
        if grid is not None:
            rh, rw = MV.refine_size(H, W, grid, vs.scale, vs.patch)
            ph, pw = rh // grid[0], rw // grid[1]
            resized.append((rh, rw, [(i, j, ph, pw) for i in range(0, rh, ph)
                                     for j in range(0, rw, pw)]))
        self.grid = grid
        self.top = base
        self.consts, self._progs, self.crops = [], [], []
        nmax = max(ch * cw for _, _, cuts in resized for _, _, ch, cw in cuts) // (p * p)
        self.raster = self._alloc(4 * nmax * KP)
        self._progs += fill_programs(cfg, self.raster, nmax, KP)    # its pad columns: zero
        for oh, ow, cuts in resized:
            pl = ResizePlan(H, W, oh, ow, "bicubic", 1 / 127.5, -1.0, clip=(0.0, 255.0))
            q, sc = pl.weights(D)
            wa, sa, ba = self._alloc(q.nbytes), self._alloc(sc.nbytes), self._alloc(
                pl.bias.nbytes)
            self.consts += [(wa, q), (sa, sc), (ba, pl.bias)]
            out = self._alloc(4 * oh * 3 * ow)
            self._progs += resize_programs(cfg, pl, src, wa, sa, ba, out)
            for y0, x0, ch, cw in cuts:
                self.crops.append((out, ow, y0, x0, ch // p, cw // p))
        self.sizes = [(h, w) for *_, h, w in self.crops]

    def _alloc(self, n: int) -> int:
        a = -(-self.top // 4096) * 4096
        self.top = a + n
        return a

    def programs(self) -> list:
        return list(self._progs)

    def crop_programs(self, k: int, dst: int) -> list:
        from ..llm import minicpmv as MV
        p, KP = self.p, self.KP
        out, ow, y0, x0, h, w = self.crops[k]
        mv = [((y0 + p * i + r) * 3 * ow + c * ow + x0, i * w * KP + c * p * p + r * p,
               w, p, p, KP) for i in range(h) for c in range(3) for r in range(p)]
        progs = blocks_programs(self.cfg, src=out, dst=self.raster, moves=mv)
        o = MV.card_order(h, w)
        return progs + blocks_programs(self.cfg, src=self.raster, dst=dst,
                                       moves=[(int(o[j]) * KP, j * KP, 1, KP, KP, KP)
                                              for j in range(h * w)])
