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
        wt, bt = self.weight(w), self.vec(b)
        ch = [(r, min(x.L, r + LINEAR_ROWS)) for r in range(0, x.L, LINEAR_ROWS)]
        for i, (r0, r1) in enumerate(ch):       # many rows: a program per row block (IMEM;
            self.op(name if len(ch) == 1 else f"{name}.{i}", dense_k,   # SmolVLA base's vision
                    x=x.rows(r0, r1), w=wt, b=bt, y=y.rows(r0, r1),     # fc1 over 1024 rows was
                    M=r1 - r0, K=x.C, act=act)                          # 8707 instructions)
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
        wt, bt = self.weight(wk), self.vec(b)
        ch = [(r, min(Lo, r + LINEAR_ROWS)) for r in range(0, Lo, LINEAR_ROWS)]
        for i, (r0, r1) in enumerate(ch):       # many rows: a program per row block (IMEM;
            v = view if len(ch) == 1 else Tensor(view.base + 4 * r0 * stride * Cin,  # Kokoro's
                                                 (r1 - r0, k * Cin), (stride * Cin, 1))  # 7321)
            self.op(name if len(ch) == 1 else f"{name}.{i}", dense_k, x=v, w=wt, b=bt,
                    y=y.rows(r0, r1), M=r1 - r0, K=k * Cin, act=act)
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

    def _chunks(self, L: int, C: int, tiles: int = 4) -> list[tuple[int, int]]:
        """Row ranges of an elementwise op whose tiles fit TMEM: the whole tensor when it fits
        (one program, as before), else blocks of rows of at most TMEM_WORDS / 4 words (two
        operand tiles and the allocator's spare rows). SmolVLA base's vision residuals
        ([1024, 768] = 786432 words) exceed r8's 131072-word TMEM in one tile. tiles: TMEM's
        share a block's tile may take (a kernel with three operands and a temporary: 6)."""
        cap = max(1, self.cfg.TMEM_WORDS // tiles)
        if L * C <= cap:
            return [(0, L)]
        rb = max(1, cap // max(1, C))
        return [(r, min(L, r + rb)) for r in range(0, L, rb)]

    def _rowwise(self, name: str, kernel, L: int, C: int, views, tiles: int = 4, **kw) -> None:
        """One op per row block: views(r0, r1) -> the kernel's tensor arguments."""
        ch = self._chunks(L, C, tiles)
        for i, (r0, r1) in enumerate(ch):
            self.op(name if len(ch) == 1 else f"{name}.{i}", kernel, **views(r0, r1), **kw)

    def unary(self, x: DT, fn: str, y: DT | None = None, name: str = "unary") -> DT:
        y = y or self.tensor(x.L, x.C, x.pad)
        self._rowwise(name, unary_k, x.L, x.C,
                      lambda r0, r1: dict(x=x.rows(r0, r1), y=y.rows(r0, r1)), fn=fn)
        return y

    def add(self, a: DT, b: DT, y: DT | None = None, name: str = "add") -> DT:
        y = y or self.tensor(a.L, a.C, a.pad)
        self._rowwise(name, binary_k, a.L, a.C,
                      lambda r0, r1: dict(a=a.rows(r0, r1), b=b.rows(r0, r1), y=y.rows(r0, r1)),
                      op="add")
        return y

    def film(self, x: DT, scale: Tensor, bias: Tensor, name: str = "film") -> DT:
        self._rowwise(name, film_k, x.L, x.C,
                      lambda r0, r1: dict(x=x.rows(r0, r1), scale=scale, bias=bias,
                                          y=x.rows(r0, r1)))
        return x

    def copy(self, x: DT, y: DT, name: str = "copy") -> DT:
        self._rowwise(name, copy_k, x.L, x.C,
                      lambda r0, r1: dict(x=x.rows(r0, r1), y=y.rows(r0, r1)))
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


def board_entries(board, words: list, progs: list, workers: int | None = None) -> list:
    """Every program's wait entries (None each without a wait-entry sequencer), computed before
    the run: once per distinct program (a denoising loop's steps repeat theirs) and in parallel.
    waits.entries alone keeps 256 (MEM_N): a SmolVLA base run (5628 prefix and 732 step
    programs, 10 steps) recomputed every step's entries on the card's critical path (406 s of
    the run's wall against 9.8 s on the card, 2026-11-19)."""
    if not board.waitc:
        return [None] * len(words)
    import os
    from . import waits as W
    D, S = (getattr(board, "_info", None) or board.info())["D"], board.slices
    keys = [W.entries_key(w, D, S, getattr(p, "run_args", None) or getattr(w, "run_args", None))
            for w, p in zip(words, progs)]
    first: dict = {}
    for i, k in enumerate(keys):
        first.setdefault(k, i)
    items = [(words[i], None, progs[i]) for i in first.values()]
    n = workers if workers is not None else min(os.cpu_count() or 1, 16)
    pool = None
    if n > 1 and len(items) > 8:
        import concurrent.futures as CF
        pool = CF.ProcessPoolExecutor(max_workers=n)
    try:
        got = W.entries_many(items, D, S, pool)
    finally:
        if pool is not None:
            pool.shutdown()
    by = dict(zip(first.keys(), got))
    return [by[k] for k in keys]


def run_board(board, progs, at: int, workers: int | None = None) -> dict:
    """The programs in order on the card, each loaded at `at` (above the image): assembled and
    their wait entries computed first (board_entries), then each loaded and run. The stats
    split the wall: prepare (assembly and entries), load (program words into the card), run
    (start to halt, the card's time and the polling)."""
    import time
    from . import isa as I
    st = {"cycles": 0, "instructions": 0, "runs": 0}
    t0 = time.perf_counter()
    ps = [p for _, p in progs]
    words = [np.asarray(I.assemble(p), np.uint32) for p in ps]
    ents = board_entries(board, words, ps, workers)
    st["distinct"] = len({w.tobytes() for w in words}) if len(words) < 100000 else None
    st["prepare_s"] = round(time.perf_counter() - t0, 2)
    t_load = t_run = 0.0
    for w, e in zip(words, ents):
        t1 = time.perf_counter()
        board.load_program(at, w, e)
        t2 = time.perf_counter()
        r = board.run(timeout=60.0)
        t_run += time.perf_counter() - t2
        t_load += t2 - t1
        st["cycles"] += r["cycles"]
        st["instructions"] += r["instructions"][0]
        st["runs"] += 1
    st["load_s"], st["run_s"] = round(t_load, 2), round(t_run, 2)
    return st



def chain_slots(words: list, ents: list, at: int, D: int, imem_words: int):
    """The programs as one chained run (docs/isa.md "HALT CHAIN"): program i's final HALT
    becomes `LI R1, slot i+1; LI R2, its instructions; HALT CHAIN R1, R2` (the last keeps its
    HALT), each in its own slot from DRAM byte `at` (D-byte aligned), with its wait entries
    after it where `ents` has them (WAITC: waits.loadable; the three control instructions
    take zero entries: none matters for them). Returns (blob: uint32 words to write at `at`,
    [(slot address, instructions)], [(words, entries)] each slot's program as loaded)."""
    from . import isa as I
    from . import waits as W
    n = len(words)
    lens = [len(w) // 8 + (2 if i + 1 < n else 0) for i, w in enumerate(words)]
    waitc = bool(ents) and ents[0] is not None
    if max(lens) > imem_words // 8:
        raise ValueError(f"a chained program of {max(lens)} instructions: IMEM holds "
                         f"{imem_words // 8}")
    # a slot's bytes: the words padded to D bytes, then (WAITC) 8 bytes of entries each
    slot = [-(-32 * k // D) * D + (8 * k if waitc else 0) for k in lens]
    addrs = [at]
    for b in slot[:-1]:
        addrs.append(addrs[-1] + -(-b // D) * D)
    out, loaded = [], []
    for i, w in enumerate(words):
        w = np.asarray(w, "<u4")
        last = w[-8:]
        if (last[0] & 0xFF) != I.HALT or (last[0] >> 24) & I.F_CHAIN:
            raise ValueError(f"program {i} does not end with a plain HALT")
        e = None if not waitc else np.asarray(ents[i], np.uint64)
        if i + 1 < n:
            tail = np.asarray(I.assemble([I.li(1, addrs[i + 1]), I.li(2, lens[i + 1]),
                                          I.halt(chain=True, ra=1, rb=2)]), "<u4")
            w = np.concatenate([w[:-8], tail])
            if waitc:
                e = np.concatenate([e[:-1], np.zeros(3, np.uint64)])
        loaded.append((w, e))
        img = W.loadable(w, e, D) if waitc else w
        pad = np.zeros(-(-4 * len(img) // D) * D // 4 - len(img), "<u4")
        out.append(np.concatenate([img, pad]))
    return np.concatenate(out), list(zip(addrs, lens)), loaded


def run_isa_chained(cfg, progs, img: np.ndarray, at: int) -> tuple[np.ndarray, int]:
    """run_isa's programs as one chained run (chain_slots, from `at`): (image, instructions,
    the chain slots' area included: [:len(img)] is the image)."""
    import dataclasses
    from . import isa as I
    from .isasim import Machine
    words = [np.asarray(I.assemble(p), np.uint32) for _, p in progs]
    blob, slots, loaded = chain_slots(words, [None] * len(words), at, cfg.D, cfg.IMEM_WORDS)
    end = at + 4 * len(blob)
    size = 1 << max(20, (end - 1).bit_length())
    c = dataclasses.replace(cfg, DRAM_BYTES=size)
    im = np.zeros(size, np.uint8)
    im[:len(img)] = img
    im[at:end] = blob.view(np.uint8)
    first = [I.Instr.decode(loaded[0][0][8 * k:8 * k + 8]) for k in range(slots[0][1])]
    s = Machine(c, [first], [im]).run(max_steps=1 << 40).slices[0]
    return s.dram[:len(img)].copy(), s.icount


def chain_segments(words: list, waitc: bool, room: int, D: int) -> list[tuple[int, int]]:
    """[lo, hi) runs of consecutive programs whose chain slots (chain_slots' sizes, a tail on
    each) fit in `room` bytes: one chained run each, the slot area reused. A single program
    larger than the room: ValueError."""
    segs, lo, used = [], 0, 0
    for i, w in enumerate(words):
        k = len(w) // 8 + 2
        b = -(-(-(-32 * k // D) * D + (8 * k if waitc else 0)) // D) * D
        if b > room:
            raise ValueError(f"program {i}'s chain slot ({b} bytes) is larger than the room "
                             f"for the slots ({room} bytes)")
        if used + b > room:
            segs.append((lo, i))
            lo, used = i, 0
        used += b
    segs.append((lo, len(words)))
    return segs


class ChainPlan:
    """A resident graph's programs prepared once for chained runs on the card (chain_slots,
    chain_segments): assembled, their wait entries computed (once per distinct program), the
    slots built. run() then costs the slots' writes (none again for a single segment: the
    runs leave the slot area above the image alone) and a start and a poll a segment, so a
    policy run again on new inputs (VLA.inputs) pays its preparation once."""

    def __init__(self, board, progs, at: int, workers: int | None = None, words=None):
        """progs: (name, program) pairs; or words: the programs already assembled (a long
        pipeline compiled in chunks, its Python programs dropped: SD 1.5's ~25k)."""
        import time
        from types import SimpleNamespace
        from . import isa as I
        t0 = time.perf_counter()
        if words is None:
            ps = [p for _, p in progs]
            words = [np.asarray(I.assemble(p), np.uint32) for p in ps]
        else:
            ps = [SimpleNamespace(run_args=None)] * len(words)
            progs = words
        ents = board_entries(board, words, ps, workers)
        info = getattr(board, "_info", None) or board.info()
        D = info["D"]
        imem = (info.get("cfg") or {}).get("IMEM_WORDS") or \
            getattr(getattr(board, "cfg", None), "IMEM_WORDS", 1 << 15)
        room = (info.get("cfg") or {}).get("DRAM_BYTES")
        waitc = bool(ents) and ents[0] is not None
        segs = chain_segments(words, waitc, room - at, D) if room else [(0, len(words))]
        self.at = at
        self.parts = [chain_slots(words[lo:hi], ents[lo:hi], at, D, imem) for lo, hi in segs]
        self.written = False
        self.stats = {"programs": len(progs), "runs": len(segs),
                      "distinct": len({w.tobytes() for w in words}),
                      "slots_mib": round(max(4 * len(b) for b, _, _ in self.parts) / 2**20, 1),
                      "prepare_s": round(time.perf_counter() - t0, 2)}

    def run(self, board, timeout: float = 600.0) -> dict:
        """Every segment in order: its slots written (a single segment's once), its first
        program loaded and started, one halt polled. The image's inputs are the caller's."""
        import time
        st = {"cycles": 0, "instructions": 0}
        load = run = 0.0
        for blob, slots, loaded in self.parts:
            t1 = time.perf_counter()
            if len(self.parts) > 1 or not self.written:
                board.write(self.at, blob.view(np.uint8))
            board.load_program(self.at, loaded[0][0], loaded[0][1])
            t2 = time.perf_counter()
            r = board.run(timeout=timeout)
            load, run = load + t2 - t1, run + time.perf_counter() - t2
            st["cycles"] += r["cycles"]
            st["instructions"] += r["instructions"][0]
        self.written = True
        st["load_s"], st["run_s"] = round(load, 2), round(run, 2)
        return st


def profile_board(board, progs, at: int, words=None, workers: int | None = None) -> list:
    """[(name, cycles)]: the programs in order on the card, each loaded and run on its own
    (run_board's way: the same order, so the same state as a chained run), its cycle count
    kept. progs: [(name, program)]; words: their assembled words when the caller has them."""
    from . import isa as I
    ps = [p for _, p in progs]
    words = words if words is not None else [np.asarray(I.assemble(p), np.uint32) for p in ps]
    ents = board_entries(board, words, ps, workers)
    out = []
    for (name, _), w, e in zip(progs, words, ents):
        board.load_program(at, w, e)
        out.append((name, board.run(timeout=60.0)["cycles"]))
    return out


def profile_table(rows, mhz: float = 133.333, top: int = 25) -> str:
    """profile_board's rows summed by op family (the name with its numbers and its trailing
    block index dropped: gen.rb01.c1.3 -> gen.rb.c): cycles, share, ms, programs."""
    import re
    fam: dict = {}
    for name, cyc in rows:
        k = re.sub(r"\.\d+$", "", name)
        k = re.sub(r"\d+", "", k)
        c, n = fam.get(k, (0, 0))
        fam[k] = (c + cyc, n + 1)
    tot = sum(c for c, _ in fam.values()) or 1
    lines = [f"{'family':<40} {'Mcycles':>9} {'share':>6} {'ms':>8} {'programs':>8}"]
    for k, (c, n) in sorted(fam.items(), key=lambda kv: -kv[1][0])[:top]:
        lines.append(f"{k:<40} {c / 1e6:9.2f} {100 * c / tot:5.1f}% {c / mhz / 1e3:8.1f} {n:8d}")
    lines.append(f"{'total':<40} {tot / 1e6:9.2f} {100.0:5.1f}% {tot / mhz / 1e3:8.1f} {len(rows):8d}")
    return "\n".join(lines)


def run_board_chained(board, progs, at: int, workers: int | None = None,
                      timeout: float = 600.0) -> dict:
    """run_board's programs as chained runs on the card (ChainPlan: prepared, then run once):
    every program written into its chain slot (chain_slots), the first loaded and started;
    the card chains through the rest (HALT CHAIN: the program loader's DRAM read each) and the
    host polls one halt. Where the slots do not fit between `at` and the DRAM's end,
    consecutive segments that do (chain_segments), one run each over the same slot area (a
    SmolVLA base: its 3.9 GiB image leaves ~130 MiB for ~400 MiB of slots: 4 runs, not
    12,948). The stats split the wall: prepare (assembly, entries, slots), load (the slots'
    writes and the first loads), run."""
    plan = ChainPlan(board, progs, at, workers)
    return {**plan.stats, **plan.run(board, timeout)}

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
    the means, then the normalization, its affine, the residual and the activation. A row
    wider than an eighth of TMEM goes in equal pieces of pixels (the SD VAE's rows at 256 px:
    256 x 256 and 128 x 512 words; TMEM's largest tile on the card is 65536 words)."""
    from .kernels import eltwise as E
    C = x.shape[1]
    cg = C // G
    Wc = W                                         # pixels a piece: W's largest divisor whose
    while Wc * C > max(C, ol.tmem_words() // 8) and Wc > 1:  # piece fits an eighth of TMEM
        Wc = max(d for d in range(1, Wc) if W % d == 0)
    pieces = [(p0, Wc) for p0 in range(0, W, Wc)]
    t = ol.empty([Wc, C])
    wv, bv = ol.load(w), ol.load(b)
    inv = 1.0 / (H * W * cg)
    sums = [ol.zeros([1]) for _ in range(G)]
    for h in ol.range(H):
        for p0, n in pieces:
            ol.load(x[h * W + p0:h * W + p0 + n, :], out=t)
            for g in range(G):
                sums[g].set(sums[g] + ol.sum(ol.sum(t[:, g * cg:(g + 1) * cg], axis=1)))
    means = [ol.zeros([1]) for _ in range(G)]
    for g in range(G):
        means[g].set(sums[g] * inv)
    sq = [ol.zeros([1]) for _ in range(G)]
    d = ol.empty([Wc, cg])
    for h in ol.range(H):
        for p0, n in pieces:
            ol.load(x[h * W + p0:h * W + p0 + n, :], out=t)
            for g in range(G):
                d.set(t[:, g * cg:(g + 1) * cg] - means[g])
                sq[g].set(sq[g] + ol.sum(ol.sumsq(d)))
    rs = [ol.zeros([1]) for _ in range(G)]
    for g in range(G):
        rs[g].set(ol.rsqrt(sq[g] * inv + eps))
    rt = ol.empty([Wc, C]) if res is not None else None
    for h in ol.range(H):
        for p0, n in pieces:
            ol.load(x[h * W + p0:h * W + p0 + n, :], out=t)
            for g in range(G):
                sl = t[:, g * cg:(g + 1) * cg]
                sl.set((sl - means[g]) * rs[g])
            t.set(t * wv[None, :] + bv[None, :])
            if res is not None:
                ol.load(res[h, p0:p0 + n, :], out=rt)
                t.set(t + rt)
            if act != "none":
                t.set(E.UNARY[act](t))
            ol.store(y[h, p0:p0 + n, :], t)


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


# ---------------------------------------------------------------- attention
@ol.jit
def attention_k(q, k, v, o, kv, H: int, KVH: int, dh: int, groups: tuple, scale: float,
                block: int = 128, kv_heads: tuple | None = None, attend: bool = True):
    """o rows = softmax(q k^T * scale) v per head (GQA), q [Lq, H*dh], k / v [Lk, KVH*dh] in
    PyTorch's [L, heads * d] rows, o [Lq, H*dh] likewise. groups: (r0, r1, n): query rows
    r0 .. r1 - 1 see keys 0 .. n - 1 (a prefix-LM / causal mask as groups of rows). The keys
    and values are quantized once into the kv scratch (sdpa's layout), each group's heads
    then attend its first n keys (flash attention, attention._attend_heads). kv_heads: the KV
    heads this program quantizes into the scratch (None: all; (): none, an earlier program's);
    attend False: the quantization only (Graph.attention splits a long attention so)."""
    from .compiler import Affine, k_scale_bytes
    from .kernels.attention import _attend_heads
    from .kernels.sdpa import up, LOG2E
    from .kernels.vit import VitKV, vit_append
    D = ol.block_size()
    dp = up(dh, D)
    Lk = k.shape[0]
    cap = up(Lk, D)
    R = ol.current().cfg.act_rows
    base = Affine.of(kv.base)
    heads, a = {}, 0
    for g in range(KVH):
        e = {}
        for name, n in (("k", cap * dp), ("ks", cap * k_scale_bytes(dp, D)), ("vt", dp * cap),
                        ("vs", 4 * cap)):
            e[name] = base + a
            a += n
        a = up(a, 64)
        heads[g] = e
    cache = VitKV(heads, cap, dp, D, 1, 0, dv=dh)
    todo = range(KVH) if kv_heads is None else kv_heads
    if len(todo):
        kt = ol.zeros([R, dp])
        vt = ol.zeros([R, dp])
        for g in todo:
            for r0 in range(0, Lk, R):
                n = min(R, Lk - r0)
                ol.load(k[r0:r0 + n, g * dh:(g + 1) * dh], out=kt[0:n, 0:dh])
                ol.load(v[r0:r0 + n, g * dh:(g + 1) * dh], out=vt[0:n, 0:dh])
                vit_append(cache, g, r0, kt[0:n, :], vt[0:n, :])
        del kt, vt
    if not attend:
        return
    G = ol.mxu_columns()
    rep = H // KVH
    sc = scale * LOG2E
    for r0g, r1g, nk in groups:
        blk = min(block, up(nk, D))
        for r0 in range(r0g, r1g, G):
            n = min(G, r1g - r0)
            qt = ol.zeros([H * n, dp])

            def qh(h, r0=r0, n=n, qt=qt):
                t = qt[h * n:(h + 1) * n, :]
                ol.load(q[r0:r0 + n, h * dh:(h + 1) * dh], out=t[:, 0:dh])
                return t
            out = ol.empty([n, H * dh])

            def emit(i, acc, den, out=out):
                out[:, i * dh:(i + 1) * dh].set(acc[:, 0:dh] * ol.recip(den)[:, None])
            _attend_heads([lambda h=h: qh(h) for h in range(H)], cache,
                          [h // rep for h in range(H)], nk, blk, sc, emit=emit)
            ol.store(o[r0:r0 + n, :], out)
            del out, qt


# Graph.attention splits an attention whose keys need more than ATTN_KV_RUNS vit_append runs
# (KV heads x row blocks), or with a group of more than ATTN_SPLIT_ROWS query rows; the query
# programs take ATTN_QBLOCK_RUNS (head, key block) attends (about 10 instructions each)
LINEAR_ROWS = 256          # Graph.linear: rows a program takes (the same rows' arithmetic)
ATTN_KV_RUNS = 96
ATTN_SPLIT_ROWS = 64
ATTN_QBLOCK_RUNS = 288


def _attn_op():
    def attention(self, q: DT, k: DT, v: DT, H: int, KVH: int, dh: int, groups,
                  scale: float | None = None, o: DT | None = None, per_prog: int = 4,
                  name: str = "attn") -> DT:
        """Masked GQA attention as programs of `per_prog` row groups each (every program
        quantizes the keys again: the scratch is its own)."""
        from .kernels.sdpa import kv_words
        o = o or self.tensor(q.L, H * dh)
        kv = self.alloc(4 * kv_words(KVH, k.L, dh, self.D))
        kvt = Tensor(kv, (kv_words(KVH, k.L, dh, self.D),), (1,))
        groups = [tuple(int(x) for x in gr) for gr in groups]
        sc = dh ** -0.5 if scale is None else scale
        args = dict(q=q.rows(), k=k.rows(), v=v.rows(), o=o.rows(), kv=kvt, H=H, KVH=KVH,
                    dh=dh, scale=sc)
        R, G, D = self.cfg.act_rows, self.cfg.MCOLS, self.D
        lim = self.cfg.IMEM_WORDS // 8
        sizes = self.__dict__.setdefault("_attn_sizes", {})

        def fits(**kw) -> bool:                 # the program's length (by shape: cached)
            key = (q.L, k.L, H, KVH, dh, tuple((r1 - r0, nk) for r0, r1, nk in kw["groups"]),
                   kw.get("kv_heads") if kw.get("kv_heads") is None else len(kw["kv_heads"]),
                   kw.get("attend", True))
            if key not in sizes:
                from .runtime import compile_kernel
                try:
                    comp, _ = compile_kernel(attention_k, self.cfg, **args, **kw)
                    sizes[key] = len(comp.programs[0])
                except ol.CompileError:
                    sizes[key] = lim + 1
            return sizes[key] <= lim

        kv_chunks = -(-k.L // R)                # vit_append runs a KV head takes
        long_rows = max(r1 - r0 for r0, r1, _ in groups) > ATTN_SPLIT_ROWS
        whole = [tuple(groups[i:i + per_prog]) for i in range(0, len(groups), per_prog)]
        if KVH * kv_chunks <= ATTN_KV_RUNS and not long_rows and all(fits(groups=g) for g in whole):
            for i, g in enumerate(whole):       # (short: each program quantizes all the keys)
                self.op(f"{name}.{i}", attention_k, **args, groups=g)
            return o
        # long (SmolVLA base's vision tower: 1024 queries and keys x 12 heads in one group, a
        # program of 154758 instructions; its text layers: 112 rows x 15 heads, 5617): the keys
        # quantized once by programs of a few KV heads, then the queries in programs of a few
        # MXU row blocks each, every one reading that scratch (the same quantization and
        # per-block arithmetic: the same results). Each program's size is measured: the heads
        # / rows a program are halved until it fits IMEM.
        hp = max(1, min(KVH, ATTN_KV_RUNS // kv_chunks))
        while hp > 1 and not fits(groups=(), kv_heads=tuple(range(hp)), attend=False):
            hp //= 2
        for i, g0 in enumerate(range(0, KVH, hp)):
            self.op(f"{name}.kv{i}", attention_k, **args, groups=(),
                    kv_heads=tuple(range(g0, min(KVH, g0 + hp))), attend=False)
        parts = []
        for r0g, r1g, nk in groups:
            kb = -(-nk // min(128, -(-nk // D) * D))           # key blocks of a query block
            qr = G * max(1, ATTN_QBLOCK_RUNS // max(1, H * kb))  # rows a program attends
            qr = min(qr, -(-(r1g - r0g) // G) * G)
            while qr > G and not fits(groups=((r0g, min(r1g, r0g + qr), nk),), kv_heads=()):
                qr = max(G, qr // 2 // G * G)
            parts += [(r, min(r1g, r + qr), nk) for r in range(r0g, r1g, qr)]
        for i, pt in enumerate(parts):
            self.op(f"{name}.{i}", attention_k, **args, groups=(pt,), kv_heads=())
        return o
    Graph.attention = attention


_attn_op()


# ---------------------------------------------------------------- transformer pieces
@ol.jit
def rope_k(x, cos, sin, y, H: int, dh: int):
    """Rotate-half RoPE per head: x [L, H*dh] (PyTorch's rows), cos / sin [L, dh / 2] (the
    rows' positions' tables): first half x1 cos - x2 sin, second x2 cos + x1 sin."""
    L = x.shape[0]
    h2 = dh // 2
    R = max(1, min(L, ol.current().cfg.act_rows))
    t = ol.empty([R, H * dh])
    for r0 in range(0, L, R):
        n = min(R, L - r0)
        v = t[0:n, :]
        ol.load(x[r0:r0 + n, :], out=v)
        c, s = ol.load(cos[r0:r0 + n, :]), ol.load(sin[r0:r0 + n, :])
        for h in range(H):
            x1 = v[:, h * dh:h * dh + h2]
            x2 = v[:, h * dh + h2:(h + 1) * dh]
            a = x1 * c - x2 * s
            x2.set(x2 * c + x1 * s)
            x1.set(a)
            del a
        ol.store(y[r0:r0 + n, :], v)
        del c, s


@ol.jit
def axpy_k(x, v, y, a: float):
    """y = x + a v (an Euler step)."""
    t = ol.load(x)
    u = ol.load(v)
    t.set(t + u * a)
    ol.store(y, t)


@ol.jit
def scale_k(x, y, a: float):
    t = ol.load(x)
    t.set(t * a)
    ol.store(y, t)


def _tf_ops():
    from .kernels import eltwise as E

    def rms_norm(self, x: DT, w, eps: float, y: DT | None = None, name="rms_norm") -> DT:
        y = y or self.tensor(x.L, x.C)
        self.op(name, E.rms_norm, x=x.rows(), w=self.vec(w), y=y.rows(), M=x.L, N=x.C, eps=eps)
        return y

    def layer_norm(self, x: DT, w, b, eps: float, y: DT | None = None, name="layer_norm") -> DT:
        y = y or self.tensor(x.L, x.C)
        self.op(name, E.layer_norm, x=x.rows(), w=self.vec(w), b=self.vec(b), y=y.rows(),
                M=x.L, N=x.C, eps=eps)
        return y

    def rope(self, x: DT, H: int, dh: int, positions, base: float = 10000.0,
             name="rope") -> DT:
        """In place (SmolVLA's apply_rope: radians = position / base^(2i / dh))."""
        pos = np.asarray(positions, np.float64)[:, None]
        ts = base ** ((2.0 / dh) * np.arange(dh // 2, dtype=np.float32))
        rad = (pos.astype(np.float32) / ts[None, :]).astype(np.float32)
        c = self.tensor(x.L, dh // 2)
        s = self.tensor(x.L, dh // 2)
        self.consts.append((c.addr, np.cos(rad).astype(np.float32)))
        self.consts.append((s.addr, np.sin(rad).astype(np.float32)))
        self.op(name, rope_k, x=x.rows(), cos=c.rows(), sin=s.rows(), y=x.rows(), H=H, dh=dh)
        return x

    def scale(self, x: DT, a: float, y: DT | None = None, name="scale") -> DT:
        y = y or x
        self._rowwise(name, scale_k, x.L, x.C,
                      lambda r0, r1: dict(x=x.rows(r0, r1), y=y.rows(r0, r1)), a=float(a))
        return y

    def axpy(self, x: DT, v: DT, a: float, name="axpy") -> DT:
        self._rowwise(name, axpy_k, x.L, x.C,
                      lambda r0, r1: dict(x=x.rows(r0, r1), v=v.rows(r0, r1), y=x.rows(r0, r1)),
                      a=float(a))
        return x

    def mul(self, a: DT, b: DT, y: DT | None = None, name="mul") -> DT:
        y = y or self.tensor(a.L, a.C)
        self._rowwise(name, binary_k, a.L, a.C,
                      lambda r0, r1: dict(a=a.rows(r0, r1), b=b.rows(r0, r1), y=y.rows(r0, r1)),
                      op="mul")
        return y

    def add_row(self, x: DT, row: Tensor, name="add_row") -> DT:
        self._rowwise(name, binary_k, x.L, x.C,
                      lambda r0, r1: dict(a=x.rows(r0, r1), b=row, y=x.rows(r0, r1)),
                      op="add", brow=True)
        return x

    def copy_view(self, src: Tensor, dst: Tensor, name="copy") -> None:
        self.op(name, copy_k, x=src, y=dst)

    Graph.rms_norm, Graph.layer_norm, Graph.rope, Graph.scale, Graph.axpy = \
        rms_norm, layer_norm, rope, scale, axpy
    Graph.mul, Graph.add_row, Graph.copy_view = mul, add_row, copy_view


_tf_ops()


@ol.jit
def copies_k(src, dst, moves: tuple, width: int):
    """Row moves in one program: (source row, destination row, destination column) of `width`
    words each, src / dst [rows, cols] views (a pixel shuffle, a prefix's assembly)."""
    t = ol.empty([1, width])
    for sr, dr, dc in moves:
        ol.load(src[sr:sr + 1, 0:width], out=t)
        ol.store(dst[dr:dr + 1, dc:dc + width], t)


def _copy_rows():
    def copy_rows(self, src: Tensor, dst: Tensor, moves, width: int, per_prog: int = 1500,
                  name="copies"):
        moves = [tuple(int(v) for v in m) for m in moves]
        for i in range(0, len(moves), per_prog):
            self.op(f"{name}.{i // per_prog}", copies_k, src=src, dst=dst,
                    moves=tuple(moves[i:i + per_prog]), width=width)
    Graph.copy_rows = copy_rows


_copy_rows()
