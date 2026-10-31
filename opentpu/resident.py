"""Resident graphs: a model's tensors live in the card's DRAM from the first op to the last, the
host only writes the inputs once, launches one program per op and reads the final result (as
cnngraph.py does for YOLOX), for models written as PyTorch modules (torchdev.py runs those op by
op through the host instead).

1-D tensors are channels-last ([L, C] rows, B = 1), with `pad` zero rows above and below the
interior that their convolutions read (the arena is zero and producers write only the interior),
so a 1-D convolution's im2col is a view: output row t reads the k * Cin contiguous words from
interior row t * stride - p, row stride stride * Cin (weights reordered k-major). A transposed
convolution writes its input into a zeroed buffer at every stride-th row and convolves with the
flipped kernel; a channel concatenation writes into column slices. GroupNorm normalizes each
group's column slice over all rows (mean, then the centered variance).

Graph.compile() traces each op into a program at fixed addresses; run_isa / run_board execute
them in order. Every op is the device's; nothing is computed on the host.
"""
from __future__ import annotations

import numpy as np

from . import language as ol
from . import quant as Q
from .compiler import QACT_KB_MAX, QTensor, Tensor

ALIGN = 4096
NC = 512


class DT:
    """A device tensor [L, C] (channels last) with `pad` zero rows above and below."""

    def __init__(self, addr: int, L: int, C: int, pad: int = 0, rs: int | None = None):
        self.addr, self.L, self.C, self.pad = addr, L, C, pad
        self.rs = C if rs is None else rs            # row stride, words

    def rows(self, r0: int = 0, r1: int | None = None, c0: int = 0, c1: int | None = None):
        r1 = self.L if r1 is None else r1
        c1 = self.C if c1 is None else c1
        return Tensor(self.addr + 4 * ((self.pad + r0) * self.rs + c0), (r1 - r0, c1 - c0),
                      (self.rs, 1))

    def cols(self, c0: int, c1: int) -> "DT":
        """Columns c0 .. c1 as a tensor of its own (a concatenation's slice)."""
        return DT(self.addr + 4 * c0, self.L, c1 - c0, self.pad, self.rs)

    @property
    def nbytes(self) -> int:
        return 4 * self.rs * (self.L + 2 * self.pad)


# ---------------------------------------------------------------- kernels (fixed addresses)
@ol.jit
def dense_k(x, w, b, y, M: int, K: int, act: str = "none", view_rows: int = 0):
    """y [M, N] = act(x [M, K] @ w^T + b): x loaded into a zero [R, Kp] tile (K padded to
    whole blocks: no over-read in the quantized operand), N in tiles of NC, K in ranges that
    fit ACT RAM."""
    from .kernels import eltwise as E
    D = ol.block_size()
    Kp = -(-K // D) * D
    N = y.shape[1]
    KC = min(ol.act_blocks(), QACT_KB_MAX) * D
    R = max(1, min(ol.current().cfg.act_rows,
                   (ol.tmem_contiguous() - 2048) // (Kp + min(N, NC) + 1)))
    bias = ol.load(b)
    xt = ol.zeros([R, Kp])
    for m0 in range(0, M, R):
        n = min(R, M - m0)
        ol.load(x[m0:m0 + n, :], out=xt[0:n, 0:K])
        for c0 in range(0, N, NC):
            c = min(NC, N - c0)
            yt = ol.empty([n, c])
            yt.set(bias[c0:c0 + c][None, :])
            for k0 in range(0, Kp, KC):
                k1 = min(Kp, k0 + KC)
                q = ol.quantize(xt[0:n, k0:k1])
                ol.dot(q, w[c0:c0 + c, k0:k1], acc=yt)
                del q
            if act != "none":
                yt.set(E.UNARY[act](yt))
            ol.store(y[m0:m0 + n, c0:c0 + c], yt)
            del yt


@ol.jit
def group_norm_k(x, w, b, y, G: int, eps: float, act: str = "none"):
    """y = act(GroupNorm(x) * w + b), x [L, C] channels last: each group's C / G columns
    normalized over all L rows."""
    from .kernels import eltwise as E
    L, C = x.shape
    cg = C // G
    t = ol.load(x)
    wv, bv = ol.load(w), ol.load(b)
    inv = 1.0 / (L * cg)
    for g in range(G):
        s = t[:, g * cg:(g + 1) * cg]
        m = ol.sum(ol.sum(s, axis=1)) * inv
        s.set(s - m)
        r = ol.rsqrt(ol.sum(ol.sumsq(s)) * inv + eps)
        s.set(s * r)
    t.set(t * wv[None, :] + bv[None, :])
    if act != "none":
        t.set(E.UNARY[act](t))
    ol.store(y, t)


@ol.jit
def binary_k(a, b, y, op: str, brow: bool = False):
    """y = a op b (b [L, C], or a [C] row vector over the rows)."""
    t = ol.load(a)
    v = ol.load(b)
    v = v[None, :] if brow else v
    t.set(t + v if op == "add" else t * v if op == "mul" else t - v)
    ol.store(y, t)


@ol.jit
def film_k(x, scale, bias, y):
    """y = x * scale + bias, scale and bias [C] over the rows (FiLM)."""
    t = ol.load(x)
    sv, bv = ol.load(scale), ol.load(bias)
    t.set(t * sv[None, :] + bv[None, :])
    ol.store(y, t)


@ol.jit
def unary_k(x, y, fn: str):
    from .kernels import eltwise as E
    t = ol.load(x)
    t.set(E.UNARY[fn](t))
    ol.store(y, t)


@ol.jit
def copy_k(x, y):
    """y = x (y a strided view: a concatenation's slice, a transposed convolution's rows)."""
    t = ol.load(x)
    ol.store(y, t)


# ---------------------------------------------------------------- graph
class Graph:
    """Ops over device tensors at fixed DRAM addresses; constants (weights) in the image."""

    def __init__(self, cfg, base: int = 0):
        self.cfg, self.D = cfg, cfg.D
        self.top = base
        self.consts: list[tuple[int, np.ndarray]] = []
        self.ops: list[tuple[str, object, dict]] = []     # (name, kernel, bound args)
        self.progs: list = []

    # ---- memory
    def alloc(self, nbytes: int) -> int:
        a = -(-self.top // ALIGN) * ALIGN
        self.top = a + nbytes
        return a

    def tensor(self, L: int, C: int, pad: int = 0) -> DT:
        t = DT(0, L, C, pad)
        t.addr = self.alloc(t.nbytes + 4 * C)          # (a row of slack)
        return t

    def const(self, a: np.ndarray) -> int:
        a = np.ascontiguousarray(a)
        at = self.alloc(a.nbytes)
        self.consts.append((at, a))
        return at

    def vec(self, v: np.ndarray) -> Tensor:
        v = np.asarray(v, np.float32).reshape(-1)
        return Tensor(self.const(v), (v.size,), (1,))

    def weight(self, w: np.ndarray) -> QTensor:
        """[N, K] float -> MX int8 rows (K padded to whole blocks) in the image."""
        from .kernels.linear import pack_weight
        p = pack_weight(np.asarray(w, np.float32), self.D)
        q, sc = Q.quantize_mxu(p, "int8", self.D)
        return QTensor.weight(self.const(q), self.const(sc), p.shape[0], p.shape[1], "int8",
                              self.D)

    def image(self, size: int | None = None) -> np.ndarray:
        img = np.zeros(size or -(-self.top // ALIGN) * ALIGN, np.uint8)
        for at, a in self.consts:
            v = a.view(np.uint8).reshape(-1)
            img[at:at + v.size] = v
        return img

    def put(self, img: np.ndarray, t: DT, x: np.ndarray) -> None:
        """An input tensor's interior rows."""
        x = np.asarray(x, np.float32).reshape(t.L, t.C)
        f = img.view(np.float32)
        for r in range(t.L):
            o = (t.addr + 4 * ((t.pad + r) * t.rs)) // 4
            f[o:o + t.C] = x[r]

    def read(self, img: np.ndarray, t: DT) -> np.ndarray:
        f = img.view(np.float32)
        return np.stack([f[(t.addr + 4 * ((t.pad + r) * t.rs)) // 4:][:t.C]
                         for r in range(t.L)])

    # ---- ops
    def op(self, name: str, kernel, **args) -> None:
        self.ops.append((name, kernel, args))

    def linear(self, x: DT, w: np.ndarray, b: np.ndarray | None, y: DT | None = None,
               act: str = "none", name: str = "linear") -> DT:
        N = w.shape[0]
        y = y or self.tensor(x.L, N)
        b = np.zeros(N, np.float32) if b is None else b
        self.op(name, dense_k, x=x.rows(), w=self.weight(w), b=self.vec(b), y=y.rows(),
                M=x.L, K=x.C, act=act)
        return y

    def conv1d(self, x: DT, w: np.ndarray, b: np.ndarray | None, stride: int = 1,
               pad: int = 0, y: DT | None = None, ypad: int = 2, act: str = "none",
               name: str = "conv1d") -> DT:
        """w [Cout, Cin, k]; x's pad >= pad (its border rows are the zero padding)."""
        Cout, Cin, k = w.shape
        if x.pad < pad or x.C != Cin or x.rs != Cin:
            raise ValueError(f"{name}: input pad {x.pad} < {pad}, or not a dense [L, {Cin}]")
        Lo = (x.L + 2 * pad - k) // stride + 1
        y = y or self.tensor(Lo, Cout, ypad)
        view = Tensor(x.addr + 4 * ((x.pad - pad) * Cin), (Lo, k * Cin), (stride * Cin, 1))
        wk = np.ascontiguousarray(w.transpose(0, 2, 1).reshape(Cout, k * Cin))
        b = np.zeros(Cout, np.float32) if b is None else b
        self.op(name, dense_k, x=view, w=self.weight(wk), b=self.vec(b), y=y.rows(), M=Lo,
                K=k * Cin, act=act)
        return y

    def conv_transpose1d(self, x: DT, w: np.ndarray, b: np.ndarray | None, stride: int,
                         pad: int, name: str = "convT") -> DT:
        """w [Cin, Cout, k] (nn.ConvTranspose1d): x written every stride-th row of a zero buffer
        with k - 1 - pad rows each side, then the flipped kernel's stride-1 convolution."""
        Cin, Cout, k = w.shape
        e = k - 1 - pad
        Lu = (x.L - 1) * stride + 1
        up = self.tensor(Lu, Cin, e)
        self.op(name + ".zero_insert", copy_k, x=x.rows(),
                y=Tensor(up.addr + 4 * (e * Cin), (x.L, Cin), (stride * Cin, 1)))
        wf = np.ascontiguousarray(w[:, :, ::-1].transpose(1, 0, 2))
        return self.conv1d(up, wf, b, 1, e, name=name)

    def group_norm(self, x: DT, G: int, w, b, eps: float, act: str = "none",
                   y: DT | None = None, name: str = "group_norm") -> DT:
        y = y or self.tensor(x.L, x.C, x.pad)
        self.op(name, group_norm_k, x=x.rows(), w=self.vec(w), b=self.vec(b), y=y.rows(),
                G=G, eps=eps, act=act)
        return y

    def unary(self, x: DT, fn: str, y: DT | None = None, name: str = "unary") -> DT:
        y = y or self.tensor(x.L, x.C, x.pad)
        self.op(name, unary_k, x=x.rows(), y=y.rows(), fn=fn)
        return y

    def add(self, a: DT, b: DT, y: DT | None = None, name: str = "add") -> DT:
        y = y or self.tensor(a.L, a.C, a.pad)
        self.op(name, binary_k, a=a.rows(), b=b.rows(), y=y.rows(), op="add")
        return y

    def film(self, x: DT, scale: Tensor, bias: Tensor, name: str = "film") -> DT:
        self.op(name, film_k, x=x.rows(), scale=scale, bias=bias, y=x.rows())
        return x

    def copy(self, x: DT, y: DT, name: str = "copy") -> DT:
        self.op(name, copy_k, x=x.rows(), y=y.rows())
        return y

    # ---- programs
    def compile(self, ops=None) -> list:
        from .runtime import compile_kernel
        out = []
        for name, kernel, args in (self.ops if ops is None else ops):
            comp, _ = compile_kernel(kernel, self.cfg, **args)
            prog = comp.programs[0]
            if len(prog) > self.cfg.IMEM_WORDS // 8:
                raise ol.CompileError(f"{name}: {len(prog)} instructions, IMEM holds "
                                      f"{self.cfg.IMEM_WORDS // 8}")
            out.append((name, prog))
        return out


def run_isa(cfg, progs, img: np.ndarray) -> tuple[np.ndarray, int]:
    """The programs in order on the ISA simulator, the image carried: (image, instructions)."""
    import dataclasses
    from .isasim import Machine
    c = dataclasses.replace(cfg, DRAM_BYTES=len(img))
    n = 0
    for _, p in progs:
        s = Machine(c, [p], [img]).run().slices[0]
        img, n = s.dram, n + s.icount
    return img, n


def run_board(board, progs, at: int) -> dict:
    """The programs in order on the card, each loaded at `at` (above the image)."""
    st = {"cycles": 0, "instructions": 0, "runs": 0}
    for _, p in progs:
        board.load(at, p)
        r = board.run(timeout=60.0)
        st["cycles"] += r["cycles"]
        st["instructions"] += r["instructions"][0]
        st["runs"] += 1
    return st


# ---------------------------------------------------------------- 2-D (NHWC) tensors
class T2:
    """A device image [H, W, C] (one sample), `border` zero pixels around it and `tail` words
    after each row (a convolution's last window reads past its row)."""

    def __init__(self, addr: int, H: int, W: int, C: int, border: int = 0, tail: int = 0):
        self.addr, self.H, self.W, self.C, self.border, self.tail = addr, H, W, C, border, tail

    @property
    def rs(self) -> int:
        return (self.W + 2 * self.border) * self.C + self.tail

    @property
    def nbytes(self) -> int:
        return 4 * self.rs * (self.H + 2 * self.border)

    def interior(self) -> Tensor:
        b = self.border
        return Tensor(self.addr + 4 * (b * self.rs + b * self.C), (self.H, self.W, self.C),
                      (self.rs, self.C, 1))

    def rows(self, pad: int, words: int) -> Tensor:
        o = self.border - pad
        if pad > self.border or o * self.C + words > self.rs:
            raise ValueError(f"a {words}-word row at pad {pad} exceeds the buffer")
        return Tensor(self.addr + 4 * (o * self.rs + o * self.C), (self.H + 2 * pad, words),
                      (self.rs, 1))

    def flat(self) -> Tensor:
        """[H * W, C] (a dense buffer: no border, no tail)."""
        if self.border or self.tail:
            raise ValueError("flat() of a padded image")
        return Tensor(self.addr, (self.H * self.W, self.C), (self.C, 1))


@ol.jit
def group_norm2d_k(x, w, b, y, H: int, W: int, G: int, eps: float, act: str = "none",
                   res=None):
    """y = act(GroupNorm(x) * w + b (+ res)): x [H*W, C] dense, y (and res) [H, W, C] views.
    Three hardware loops over the image rows: the groups' sums, their squared deviations from
    the means, then the normalization, its affine, the residual and the activation."""
    from .kernels import eltwise as E
    C = x.shape[1]
    cg = C // G
    t = ol.empty([W, C])
    wv, bv = ol.load(w), ol.load(b)
    inv = 1.0 / (H * W * cg)
    sums = [ol.zeros([1]) for _ in range(G)]
    for h in ol.range(H):
        ol.load(x[h * W:(h + 1) * W, :], out=t)
        for g in range(G):
            sums[g].set(sums[g] + ol.sum(ol.sum(t[:, g * cg:(g + 1) * cg], axis=1)))
    means = [ol.zeros([1]) for _ in range(G)]
    for g in range(G):
        means[g].set(sums[g] * inv)
    sq = [ol.zeros([1]) for _ in range(G)]
    d = ol.empty([W, cg])
    for h in ol.range(H):
        ol.load(x[h * W:(h + 1) * W, :], out=t)
        for g in range(G):
            d.set(t[:, g * cg:(g + 1) * cg] - means[g])
            sq[g].set(sq[g] + ol.sum(ol.sumsq(d)))
    rs = [ol.zeros([1]) for _ in range(G)]
    for g in range(G):
        rs[g].set(ol.rsqrt(sq[g] * inv + eps))
    rt = ol.empty([W, C]) if res is not None else None
    for h in ol.range(H):
        ol.load(x[h * W:(h + 1) * W, :], out=t)
        for g in range(G):
            sl = t[:, g * cg:(g + 1) * cg]
            sl.set((sl - means[g]) * rs[g])
        t.set(t * wv[None, :] + bv[None, :])
        if res is not None:
            ol.load(res[h, 0:W, :], out=rt)
            t.set(t + rt)
        if act != "none":
            t.set(E.UNARY[act](t))
        ol.store(y[h, 0:W, :], t)


@ol.jit
def spatial_softmax_k(x, px: tuple, py: tuple, y):
    """LeRobot's SpatialSoftmax on [HW, K] scores (pixels x keypoints): per keypoint (column)
    the softmax over the pixels (rows), then y [1, 2K] = (expected x of each keypoint, then
    expected y of each) -- the caller's next linear takes its columns in that order."""
    HW, K = x.shape
    t = ol.load(x)
    m = t[0:1, :]
    mx = ol.empty([1, K])
    mx.set(m)
    for r in range(1, HW):
        mx.set(ol.maximum(mx, t[r:r + 1, :]))
    for r in range(HW):
        t[r:r + 1, :].set(ol.exp2((t[r:r + 1, :] - mx) * ol.LOG2E))
    s = ol.empty([1, K])
    sx = ol.empty([1, K])
    sy = ol.empty([1, K])
    s.set(t[0:1, :])
    sx.set(t[0:1, :] * float(px[0]))
    sy.set(t[0:1, :] * float(py[0]))
    for r in range(1, HW):
        s.set(s + t[r:r + 1, :])
        sx.set(sx + t[r:r + 1, :] * float(px[r]))
        sy.set(sy + t[r:r + 1, :] * float(py[r]))
    inv = ol.recip(s)
    o = ol.empty([1, 2 * K])
    o[:, 0:K].set(sx * inv)
    o[:, K:2 * K].set(sy * inv)
    ol.store(y, o)


def _graph_2d_ops():
    def tensor2(self, H, W, C, border=0, tail=0) -> T2:
        t = T2(0, H, W, C, border, tail)
        t.addr = self.alloc(t.nbytes + 4 * C)
        return t

    def conv2d(self, x: T2, w: np.ndarray, b, stride: int, pad: int, name="conv2d") -> T2:
        """A dense [OH*OW, Cout] output (its GroupNorm writes the padded consumer's buffer)."""
        from .kernels import conv as C
        Cout, Cin, KH, KW = w.shape
        OH, OW, Hp, Wp, Kp, rw = C.geometry(x.H, x.W, Cin, KH, KW, stride, pad, self.D)
        y = self.tensor2(OH, OW, Cout)
        p = C.pack_conv_weight(np.asarray(w, np.float32), self.D)
        q, sc = Q.quantize_mxu(p, "int8", self.D)
        wq = QTensor.weight(self.const(q), self.const(sc), p.shape[0], p.shape[1], "int8",
                            self.D)
        b = np.zeros(Cout, np.float32) if b is None else b
        self.op(name, C.conv2d, x=x.rows(pad, rw), w=wq, b=self.vec(b), y=y.interior(),
                H=x.H, W=x.W, Cin=Cin, Cout=Cout, KH=KH, KW=KW, stride=stride, pad=pad,
                act="none")
        return y

    def group_norm2d(self, x: T2, G, w, b, eps, act="none", res: T2 | None = None,
                     border=1, tail=None, name="gn2d") -> T2:
        y = self.tensor2(x.H, x.W, x.C, border, self.D if tail is None else tail)
        self.op(name, group_norm2d_k, x=x.flat(), w=self.vec(w), b=self.vec(b),
                y=y.interior(), H=x.H, W=x.W, G=G, eps=eps, act=act,
                res=None if res is None else res.interior())
        return y

    def maxpool2d(self, x: T2, K: int, stride: int, pad: int, name="maxpool") -> T2:
        """(x >= 0: a ReLU's output, so the zero border is max pooling's -inf)."""
        from .kernels import conv as C
        OH = (x.H + 2 * pad - K) // stride + 1
        OW = (x.W + 2 * pad - K) // stride + 1
        y = self.tensor2(OH, OW, x.C, 1, self.D)
        self.op(name, C.maxpool2d, x=x.rows(pad, (x.W + 2 * pad) * x.C), y=y.interior(),
                H=x.H, W=x.W, C=x.C, K=K, stride=stride, pad=pad)
        return y

    Graph.tensor2, Graph.conv2d, Graph.group_norm2d, Graph.maxpool2d = \
        tensor2, conv2d, group_norm2d, maxpool2d


_graph_2d_ops()
