"""PyTorch modules on openTPU: a model's nn.Linear layers run as device kernels
(kernels/linear.py) on the ISA simulator or the card, the rest of its graph in PyTorch on the
host, every multiply-accumulate counted by where it ran (Coverage).

This is the first stage of bringing a model graph to the device (as tools/yolox_run.py was for
YOLOX before opentpu/cnngraph.py): it runs a real checkpoint's own forward, so its numerics are
checked against the framework at once, and it reports exactly which work is still on the host.
Each call writes its activations and the layer's weights (int8 MX, quantized once per layer)
and runs one program; weights are not yet resident across calls (a resident image is the next
stage). Nothing here falls back silently: a layer the kernel cannot take raises.
"""
from __future__ import annotations

import collections
import time

import numpy as np

PROG_GAP = 1 << 20


class ProgramTooLong(RuntimeError):
    """A kernel's program is longer than IMEM holds (the board's loader takes IMEM_WORDS / 8
    instructions; a longer one would run overwritten -- SmolVLA base's linears on r4, rel L2 0.88
    against float, before the loader refused it). n: its instructions, cap: IMEM's."""

    def __init__(self, name: str, n: int, cap: int):
        super().__init__(f"{name}: a program of {n} instructions, IMEM holds {cap}")
        self.n, self.cap = n, cap


def run_on_board(board, comp, img, out: str = "y"):
    """Run one compiled kernel on the card: its DRAM image (up to its last placed byte) at 0, the
    program above it, then the output read back. Returns (output, run stats)."""
    from . import quant as Q
    from .runtime import Input, Output, Weight, collect
    D = comp.cfg.D
    top = 0                     # the end of every placed argument: all of it is written, so the
    for a, addr in comp.layout.params.values():     # card never reads a beat it has not written
        if isinstance(a, Output):
            top = max(top, addr + 4 * int(np.prod(a.shape)))
        elif isinstance(a, Weight):
            n, k = a.array.shape
            top = max(top, addr[0] + n * Q.row_bytes(k, a.fmt, D),
                      addr[1] + n * Q.scale_bytes(k, a.fmt, D))
        elif isinstance(a, Input):
            top = max(top, addr + a.array.nbytes)
        else:
            raise NotImplementedError(f"placing {type(a).__name__} on the card")
    top = -(-top // 4096) * 4096
    board.write(0, img[:top].copy())
    board.load(top + PROG_GAP, comp.programs[0])
    st = board.run(timeout=60.0)
    dram = img.copy()
    for a, addr in comp.layout.params.values():
        if isinstance(a, Output):
            n = 4 * int(np.prod(a.shape))
            dram[addr:addr + n] = board.read(addr, n)
    return collect(comp, [dram], [None], st).outputs[out], st


class Runner:
    """Runs kernels on `backend`: "isa" (the ISA simulator) or "board" (the card at `dev`, its
    bitstream's Config); check_isa: every board run also on the ISA simulator from the same
    image, outputs required bit-identical. resident (default): a Weight argument is quantized
    and written to DRAM once, the first time it is used, and stays there (runtime.Resident);
    each call then moves only its activations, outputs and program. resident=False: every
    call's image carries its weights, as before. (SmolVLA base on r4: 2.35 G card cycles,
    ~18 s, in ~1415 s of device calls; per call the host also compiles the kernel and computes
    its wait entries, tens to ~100 ms each -- weights are one part of that overhead.)"""

    def __init__(self, backend: str = "isa", cfg=None, dev: str = "/dev/xdma0",
                 check_isa: bool = False, resident: bool = True, cache_programs: bool = True):
        self.backend, self.check_isa = backend, check_isa
        self.board = None
        if backend == "board":
            from .host.board import Board, XdmaTransport, device_config
            self.board = Board(XdmaTransport(dev), check=False)
            cfg = cfg or device_config(self.board.info())
        elif backend != "isa":
            raise ValueError(f"backend {backend}: isa or board")
        if cfg is None:
            from .isasim import board_config
            cfg = board_config(DRAM_BYTES=1 << 28)
        self.cfg = cfg
        self.stats = collections.Counter()
        self.cycles = 0
        self.device_s = 0.0
        self.mismatch = 0
        # resident weights: quantized and written once, top-down from the end of DRAM; each
        # call places only its activations and outputs below them (and its program above those)
        self.resident = resident and cfg.S == 1
        self._heap = cfg.DRAM_BYTES
        self._res = {}                  # id(Weight) -> (Resident, the Weight: kept alive)
        self._regions = []              # (addr, bytes): what the ISA simulator's images get
        self.cache_programs = cache_programs
        self._progs = {}                # call key -> (the programs as text, their Compiled)
        self.weight_bytes = 0           # written once
        self.call_bytes = 0             # written per call (activations, programs)

    def _resident_of(self, w):
        """w's Resident: its quantized rows and scales written once (the card, or kept for the
        ISA simulator's images), placed below the previous ones."""
        from . import quant as Q
        from .runtime import ALIGN, Resident
        r = self._res.get(id(w))
        if r is not None:
            return r[0]
        D = self.cfg.D
        N, K = w.array.shape
        q, sc = Q.quantize_mxu(w.array, w.fmt, D)
        q = np.ascontiguousarray(q).view(np.uint8).reshape(-1)
        sc = np.ascontiguousarray(sc).view(np.uint8).reshape(-1)
        saddr = (self._heap - sc.size) // ALIGN * ALIGN
        daddr = (saddr - q.size) // ALIGN * ALIGN
        floor = min(self.MIN_CALL_BYTES, self.cfg.DRAM_BYTES // 2)
        if daddr < floor:
            raise MemoryError(f"resident weights: {self.weight_bytes + q.size + sc.size} bytes "
                              f"leave less than {floor} for a call's data")
        self._heap = daddr
        if self.board is not None:
            self.board.write(daddr, q.copy())
            self.board.write(saddr, sc.copy())
        if self.board is None or self.check_isa:    # the ISA simulator's images get them too
            self._regions += [(daddr, q), (saddr, sc)]
        self.weight_bytes += q.size + sc.size
        res = Resident(daddr, saddr, N, K, w.fmt)
        self._res[id(w)] = (res, w)
        return res

    def _key(self, kernel, limit, args):
        """A call's program cache key: the kernel and each argument's shape (an
        Input / Output), placement (a Resident) or value (anything else); None when an argument
        has no key (a non-resident Weight: its bytes are in the image, the program is not)."""
        from .runtime import Input, Output, Resident
        parts = [kernel.__name__, id(kernel)]          # (the limit only bounds placement:
        #                                                 the same args place at the same addresses)
        for k in sorted(args):
            v = args[k]
            if isinstance(v, Input):
                parts.append((k, "in", v.array.shape))
            elif isinstance(v, Output):
                parts.append((k, "out", v.shape))
            elif isinstance(v, Resident):
                parts.append((k, "res", v.daddr, v.saddr, v.N, v.K, v.fmt))
            elif isinstance(v, (int, float, str, bool, type(None))):
                parts.append((k, v))
            elif isinstance(v, tuple) and all(isinstance(x, (int, float, str, bool)) for x in v):
                parts.append((k, v))
            else:
                return None
        return tuple(parts)

    def _compile(self, kernel, limit, args):
        """compile_kernel, with each distinct call's programs traced once: a call alike in
        every key part (the same layer at the same shapes, its weights resident) places its
        data and reuses the programs and store records (with this call's Output objects).
        Tracing a linear took 37-88 ms a call on the host (2026-10-24); SmolVLA's Euler steps
        repeat the same calls. OTPU_PROG_CACHE_CHECK=1 traces anyway and compares."""
        import os
        from .compiler import Compiled
        from .runtime import Output, compile_kernel, place
        key = self._key(kernel, limit, args) if self.cache_programs else None
        hit = self._progs.get(key) if key is not None else None
        if hit is None or os.environ.get("OTPU_PROG_CACHE_CHECK") == "1":
            comp, imgs = compile_kernel(kernel, self.cfg, _dram_limit=limit, **args)
            if hit is not None and [list(map(str, p)) for p in comp.programs] != hit[0]:
                raise RuntimeError(f"{kernel.__name__}: the cached program differs from a fresh "
                                   "trace (a kernel that reads its inputs' data while tracing)")
            if key is not None and hit is None:
                self._progs[key] = ([list(map(str, p)) for p in comp.programs], comp)
                self.stats["traced"] += 1
            return comp, imgs
        comp0 = hit[1]
        dram, layout, _ = place(self.cfg, args, limit)
        # the store records name the Output objects of the call that traced them: this call's
        new_of = {}
        for name, (a, _) in comp0.layout.params.items():
            if isinstance(a, Output):
                new_of[id(a)] = layout.params[name][0]
        stores = [[(new_of.get(id(h), h), off, n) for h, off, n in recs] for recs in comp0.stores]
        self.stats["cached"] += 1
        return Compiled(self.cfg, comp0.programs, stores, layout), dram.images

    def rows(self, kernel, xs: dict, N: int, vecs: dict | None = None, **kw) -> np.ndarray:
        """A row-wise kernel (kernels/eltwise.py) over [M, N] arrays xs (by argument name, the
        same M), vecs [N] arrays, kw its other arguments: y [M, N], the rows in chunks whose
        programs fit IMEM."""
        from .runtime import Input, Output
        M = next(iter(xs.values())).shape[0]
        out = np.empty((M, N), np.float32)
        steps = self.__dict__.setdefault("_row_steps", {})
        m0, step = 0, steps.get((kernel.__name__, N), M)
        while m0 < M:
            n = min(step, M - m0)
            args = {k: Input(np.ascontiguousarray(v[m0:m0 + n])) for k, v in xs.items()}
            args.update({k: Input(v) for k, v in (vecs or {}).items()})
            try:
                out[m0:m0 + n] = self.run(kernel, y=Output((n, N)), M=n, N=N, **args, **kw)
            except ProgramTooLong as e:
                if n <= 1:
                    raise
                step = max(1, min(n - 1, n * e.cap // e.n) // 16 * 16 or 1)
                steps[(kernel.__name__, N)] = step
                continue
            m0 += n
        return out

    MIN_CALL_BYTES = 64 << 20   # DRAM kept for a call's data and program (half a small DRAM)

    def run(self, kernel, out: str = "y", **args) -> np.ndarray:
        from .runtime import Weight, compile_kernel, run
        limit = None
        if self.resident:
            args = {k: (self._resident_of(v) if isinstance(v, Weight) and v.shard is None else v)
                    for k, v in args.items()}
            # the call's data, then the program PROG_GAP above it, all below the weights
            limit = self._heap - PROG_GAP - 4 * self.cfg.IMEM_WORDS - 4096
        comp, imgs = self._compile(kernel, limit, args)
        for addr, b in self._regions:           # the ISA simulator: the resident weights too
            for img in imgs:                    # (on the card they are already there; the
                img[addr:addr + b.size] = b     # card's image is written only up to its top)
        from . import quant as Q
        from .runtime import Input
        D = self.cfg.D
        self.call_bytes += 32 * sum(len(p) for p in comp.programs)     # 8 words an instruction
        for a, _ in comp.layout.params.values():
            if isinstance(a, Input):
                self.call_bytes += a.array.nbytes
            elif isinstance(a, Weight):                                 # as quantized and sent
                n, k = a.array.shape
                self.call_bytes += n * (Q.row_bytes(k, a.fmt, D) + Q.scale_bytes(k, a.fmt, D))
        cap = self.cfg.IMEM_WORDS // 8
        n = max(len(p) for p in comp.programs)
        if n > cap:
            raise ProgramTooLong(kernel.__name__, n, cap)
        t0 = time.perf_counter()
        if self.board is None:
            y = run(comp, imgs, "isa").outputs[out]
        else:
            y, st = run_on_board(self.board, comp, imgs[0], out)
            self.cycles += int((st or {}).get("cycles") or 0)
            if self.check_isa:
                z = run(comp, imgs, "isa").outputs[out]
                if not np.array_equal(y.view(np.uint32), z.view(np.uint32)):
                    self.mismatch += 1
                    raise RuntimeError(f"{kernel.__name__}: card != ISA simulator "
                                       f"(max |d| {float(np.abs(y - z).max())})")
        self.device_s += time.perf_counter() - t0
        self.stats["runs"] += 1
        return y


class Coverage:
    """Multiply-accumulates by where they ran: "device" (kernels) and "host" (PyTorch ops of
    the matmul family seen while the model ran), by op name."""

    def __init__(self):
        self.macs = collections.Counter()
        self.calls = collections.Counter()
        self.elems = collections.Counter()      # host elementwise / reduction ops: outputs
        self.delems = collections.Counter()     # device elementwise / norm ops: outputs

    def add(self, where: str, op: str, macs: int) -> None:
        self.macs[where, op] += int(macs)
        self.calls[where, op] += 1

    def total(self, where: str) -> int:
        return sum(v for (w, _), v in self.macs.items() if w == where)

    def report(self) -> str:
        d, h = self.total("device"), self.total("host")
        lines = [f"MACs on the device {d:,} ({100 * d / max(d + h, 1):.2f}%), host {h:,}"]
        for (w, op), v in sorted(self.macs.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {w:6s} {op:28s} {v:>16,} MACs in {self.calls[w, op]} calls")
        if self.delems:
            top = ", ".join(f"{k} {v:,}" for k, v in self.delems.most_common(8))
            lines.append(f"device elementwise / norm outputs {sum(self.delems.values()):,} "
                         f"elements ({top})")
        if self.elems:
            top = ", ".join(f"{k} {v:,}" for k, v in self.elems.most_common(8))
            lines.append(f"host elementwise / norm / reduction outputs {sum(self.elems.values()):,}"
                         f" elements (no MACs; {top})")
        return "\n".join(lines)


def host_counter(cov: Coverage):
    """A TorchFunctionMode counting the MACs of matmul-family ops PyTorch runs on the host
    (matmul / bmm / mm / einsum / linear / conv / scaled_dot_product_attention)."""
    import torch
    from torch.overrides import TorchFunctionMode

    def numel(t):
        return int(t.numel()) if isinstance(t, torch.Tensor) else 0

    moves = {"reshape", "view", "transpose", "permute", "expand", "to", "contiguous", "clone",
             "__getitem__", "__setitem__", "split", "chunk", "stack", "cat", "size", "dim",
             "flatten", "unsqueeze", "squeeze", "repeat", "expand_as", "view_as", "type",
             "float", "bfloat16", "half", "detach", "cpu", "zeros_like", "ones_like",
             "empty_like", "narrow", "select", "unbind", "masked_fill", "where", "copy_",
             "__get__", "numel", "shape", "item", "tolist", "__len__", "__eq__", "__ne__",
             "__lt__", "__gt__", "__le__", "__ge__", "__bool__", "__and__", "__or__",
             "__invert__", "arange", "full", "zeros", "ones", "empty", "tensor", "from_numpy",
             "cumsum", "sum_bool", "long", "bool", "int", "argmax", "max", "min"}

    class Count(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = getattr(func, "__name__", str(func))
            try:
                if name in ("matmul", "bmm", "mm", "__matmul__") and len(args) >= 2:
                    a, b = args[0], args[1]
                    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                        out = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2]) \
                            if a.dim() > 2 and b.dim() > 2 else ()
                        m = a.shape[-2] if a.dim() >= 2 else 1
                        n = b.shape[-1] if b.dim() >= 2 else 1
                        cov.add("host", name, int(np.prod(out)) * m * a.shape[-1] * n)
                elif name == "linear":
                    x, w = args[0], args[1]
                    cov.add("host", "linear", numel(x) // x.shape[-1] * w.numel())
                elif name == "scaled_dot_product_attention":
                    q, k = args[0], args[1]
                    cov.add("host", "sdpa", 2 * numel(q) * k.shape[-2])
                elif name == "einsum":
                    cov.add("host", "einsum", 0)        # counted below by the inner op
                elif name in ("conv1d", "conv2d"):
                    x, w = args[0], args[1]
                    out = func(*args, **kwargs)
                    cov.add("host", name, out.numel() // out.shape[1] * w.numel())
                    return out
            except Exception:                       # counting must never change the run
                pass
            out = func(*args, **kwargs)
            try:
                if name not in moves and name not in ("matmul", "bmm", "mm", "__matmul__",
                                                      "linear", "scaled_dot_product_attention",
                                                      "einsum") \
                        and isinstance(out, torch.Tensor) and out.is_floating_point():
                    cov.elems[name] += out.numel()
            except Exception:
                pass
            return out
    return Count()


class DeviceLinear:
    """Replaces an nn.Linear's forward: y = x W^T + b on the device (kernels/linear.py).
    wsplit: int8 hi + lo weights (conv.split_weight; two MMs)."""

    def __init__(self, lin, runner: Runner, cov: Coverage, name: str = "", wsplit: bool = False):
        from .kernels import linear as L
        from .runtime import Weight
        D = runner.cfg.D
        w = lin.weight.detach().float().cpu().numpy()
        self.N, self.K = w.shape
        self.b = (lin.bias.detach().float().cpu().numpy() if lin.bias is not None
                  else np.zeros(self.N, np.float32))
        packed = L.pack_weight(w, D)
        if wsplit:
            from .kernels.conv import split_weight
            hi, lo = split_weight(packed, D)
            self.w, self.wlo = Weight(hi, fmt="int8"), Weight(lo, fmt="int8")
        else:
            self.w, self.wlo = Weight(packed, fmt="int8"), None
        self.runner, self.cov, self.name = runner, cov, name

    def __call__(self, x):
        import torch
        from .kernels import linear as L
        from .runtime import Input, Output
        shp, dt = x.shape, x.dtype
        a = x.detach().float().reshape(-1, self.K).cpu().numpy()
        M = a.shape[0]
        if M == 0:
            return torch.zeros(*shp[:-1], self.N, dtype=dt)
        kw = dict(wlo=self.wlo) if self.wlo is not None else {}
        D = self.runner.cfg.D
        # the kernel's program grows with M (a row tile at a time, unrolled): rows in chunks of
        # `step` (a multiple of D: each chunk's tiles and their activation scales are the whole
        # run's, so the result is the same bit for bit) small enough for IMEM
        y = np.empty((M, self.N), np.float32)
        m0, step = 0, getattr(self, "_step", None) or M
        while m0 < M:
            n = min(step, M - m0)
            try:
                y[m0:m0 + n] = self.runner.run(
                    L.linear, x=Input(L.pad_rows(a[m0:m0 + n], D)), w=self.w, b=Input(self.b),
                    y=Output((n, self.N)), M=n, K=self.K, N=self.N, **kw)
            except ProgramTooLong as e:
                if n <= D:
                    raise
                step = max(D, min(n - 1, n * e.cap // e.n) // D * D)
                self._step = step
                continue
            m0 += n
        self.cov.add("device", "linear", M * self.K * self.N)
        return torch.from_numpy(np.ascontiguousarray(y)).reshape(*shp[:-1], self.N).to(dt)


class DeviceConv2d:
    """Replaces an nn.Conv2d's forward (groups 1, square kernel and stride, zeros padding, no
    dilation) with kernels/conv.conv2d on the device, batch by batch (NCHW in and out)."""

    def __init__(self, conv, runner: Runner, cov: Coverage, name: str = "",
                 wsplit: bool = False):
        from .kernels import conv as C
        from .runtime import Weight
        pad = (0, 0) if conv.padding == "valid" else conv.padding
        if conv.groups != 1 or conv.dilation != (1, 1) or conv.padding_mode != "zeros" \
                or conv.kernel_size[0] != conv.kernel_size[1] \
                or conv.stride[0] != conv.stride[1] or isinstance(pad, str) \
                or pad[0] != pad[1]:
            raise NotImplementedError(f"{name}: conv {conv} not on the device kernel")
        D = runner.cfg.D
        w = conv.weight.detach().float().cpu().numpy()
        self.Cout, self.Cin, self.K, _ = w.shape
        self.s, self.p = conv.stride[0], pad[0]
        self.b = (conv.bias.detach().float().cpu().numpy() if conv.bias is not None
                  else np.zeros(self.Cout, np.float32))
        packed = C.pack_conv_weight(w, D)
        if wsplit:
            hi, lo = C.split_weight(packed, D)
            self.w, self.wlo = Weight(hi, fmt="int8"), Weight(lo, fmt="int8")
        else:
            self.w, self.wlo = Weight(packed, fmt="int8"), None
        self.runner, self.cov, self.name = runner, cov, name

    def __call__(self, x):
        import torch
        from .kernels import conv as C
        from .runtime import Input, Output
        D, dt = self.runner.cfg.D, x.dtype
        outs = []
        for xb in x.detach().float().cpu().numpy():
            _, H, W = xb.shape
            OH, OW, *_ = C.geometry(H, W, self.Cin, self.K, self.K, self.s, self.p, D)
            kw = dict(wlo=self.wlo) if self.wlo is not None else {}
            y = self.runner.run(C.conv2d, x=Input(C.pad_input(xb.transpose(1, 2, 0), self.K,
                                                              self.K, self.s, self.p, D)),
                                w=self.w, b=Input(self.b), y=Output((OH * OW, self.Cout)), H=H,
                                W=W, Cin=self.Cin, Cout=self.Cout, KH=self.K, KW=self.K,
                                stride=self.s, pad=self.p, act="none", **kw)
            self.cov.add("device", "conv2d", OH * OW * self.Cout * self.Cin * self.K * self.K)
            outs.append(y.reshape(OH, OW, self.Cout).transpose(2, 0, 1))
        return torch.from_numpy(np.ascontiguousarray(np.stack(outs))).to(dt)


def key_prefixes(mask):
    """A boolean attention mask [Lq, Lk] whose every row sees a prefix of the keys, as
    [(n, rows)]: the distinct prefix lengths and the query rows of each; None if some row's
    keys are not a prefix (then the mask is not one DeviceAttention runs)."""
    m = np.asarray(mask, bool)
    n = m.sum(1)
    pre = np.arange(m.shape[1])[None, :] < n[:, None]
    if not np.array_equal(m, pre):
        return None
    return [(int(k), np.nonzero(n == k)[0]) for k in np.unique(n)]


class DeviceAttention:
    """Masked multi-head attention on the device (kernels/sdpa.py) for masks whose rows each
    see a prefix of the keys -- SmolVLA's prefix-LM, causal-block and cross-attention masks:
    one device run per distinct prefix, its query rows against those keys. Called as SmolVLA's
    eager_attention_forward(attention_mask [B, Lq, Lk], batch_size, head_dim, q [B, Lq, H, d],
    k [B, Lk, KVH, d], v [B, Lk, KVH, d]) -> [B, Lq, H * d]. A mask of another form raises
    (allow_host: it runs on the host instead, counted there)."""

    def __init__(self, runner: Runner, cov: Coverage, n_heads: int, n_kv_heads: int,
                 host_fn=None, allow_host: bool = False):
        self.runner, self.cov, self.H, self.KVH = runner, cov, n_heads, n_kv_heads
        self.host_fn, self.allow_host = host_fn, allow_host

    def hf(self, module, query, key, value, attention_mask, dropout: float = 0.0,
           scaling: float | None = None, **kwargs):
        """As a transformers attention function (AttentionInterface: query [B, H, Lq, d], key /
        value [B, KVH, Lk, d], an additive or boolean mask or None) -> ([B, Lq, H, d], None)."""
        import torch
        B, H, Lq, d = query.shape
        Lk = key.shape[2]
        if attention_mask is None:
            m = torch.ones(B, Lq, Lk, dtype=torch.bool)
        else:
            am = attention_mask[:, 0] if attention_mask.dim() == 4 else attention_mask
            m = am if am.dtype == torch.bool else (am == 0)
            m = m.expand(B, Lq, Lk)
        y = self(m, B, d, query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                 heads=(H, key.shape[1]), scale=scaling)
        return y.reshape(B, Lq, H, d), None

    def __call__(self, attention_mask, batch_size, head_dim, query_states, key_states,
                 value_states, heads=None, scale=None):
        import torch
        from .kernels import sdpa as A
        from .runtime import Input, Output
        D, dv = self.runner.cfg.D, head_dim
        H, KVH = heads or (self.H, self.KVH)
        scale = dv ** -0.5 if scale is None else float(scale)
        dp = A.up(dv, D)
        outs = []
        for b in range(batch_size):
            mask = attention_mask[b].detach().cpu().numpy()
            segs = key_prefixes(mask)
            if segs is None:
                if not self.allow_host or self.host_fn is None:
                    raise NotImplementedError("attention mask rows are not key prefixes")
                return self.host_fn(attention_mask, batch_size, head_dim, query_states,
                                    key_states, value_states)
            q = query_states[b].detach().float().cpu().numpy()      # [Lq, H, d]
            k = key_states[b].detach().float().cpu().numpy()        # [Lk, KVH, d]
            v = value_states[b].detach().float().cpu().numpy()
            Lq = q.shape[0]
            o = np.zeros((Lq, H * dv), np.float32)
            for n, rows in segs:
                if n == 0:
                    raise NotImplementedError("a query row that sees no key")
                qq = np.zeros((H, len(rows), dp), np.float32)
                qq[:, :, :dv] = q[rows].transpose(1, 0, 2)
                kk = np.zeros((KVH, n, dp), np.float32)
                vv = np.zeros((KVH, n, dp), np.float32)
                kk[:, :, :dv] = k[:n].transpose(1, 0, 2)
                vv[:, :, :dv] = v[:n].transpose(1, 0, 2)
                y = self.runner.run(A.sdpa, out="o", q=Input(qq.reshape(-1, dp)),
                                    k=Input(kk.reshape(-1, dp)), v=Input(vv.reshape(-1, dp)),
                                    o=Output((len(rows), H * dv)),
                                    kv=Input(np.zeros(A.kv_words(KVH, n, dv, D), np.float32)),
                                    H=H, KVH=KVH, Lq=len(rows), Lk=n, dv=dv,
                                    scale=scale)
                o[rows] = y
                self.cov.add("device", "attention", 2 * H * len(rows) * n * dv)
            outs.append(o)
        return torch.from_numpy(np.stack(outs)).to(query_states.dtype)


def hf_attention(model, att: DeviceAttention, name: str = "opentpu") -> int:
    """Every transformers sub-model of `model` whose config names an attention implementation
    runs its attention through `att` (registered as transformers attention function `name`).
    Returns the number of configs switched."""
    from transformers import AttentionInterface
    AttentionInterface.register(name, att.hf)
    n = 0
    for m in model.modules():
        c = getattr(m, "config", None)
        if c is not None and getattr(c, "_attn_implementation", None) not in (None, name):
            c._attn_implementation = name
            n += 1
    return n


class DeviceRows:
    """Replaces a row-wise module's forward (an activation, LayerNorm, RMSNorm over the last
    dimension) with kernels/eltwise.py on the device: the input's leading dimensions flattened
    into rows, the result in the input's dtype."""

    def __init__(self, runner: Runner, cov: Coverage, name: str, kind: str, fn: str = "",
                 w=None, b=None, eps: float = 0.0):
        self.runner, self.cov, self.name, self.kind, self.fn = runner, cov, name, kind, fn
        self.w = None if w is None else w.detach().float().cpu().numpy().copy()
        self.b = None if b is None else b.detach().float().cpu().numpy().copy()
        self.eps = float(eps)

    def __call__(self, x, *args, **kwargs):
        import torch
        from .kernels import eltwise as E
        N, dt = x.shape[-1], x.dtype
        a = x.detach().float().reshape(-1, N).cpu().numpy()
        if self.kind == "act":
            y = self.runner.rows(E.unary, {"x": a}, N, fn=self.fn)
        elif self.kind == "layer_norm":
            y = self.runner.rows(E.layer_norm, {"x": a}, N, {"w": self.w, "b": self.b},
                                 eps=self.eps)
        else:
            y = self.runner.rows(E.rms_norm, {"x": a}, N, {"w": self.w}, eps=self.eps)
        self.cov.delems[self.fn or self.kind] += y.size
        return torch.from_numpy(y).reshape(x.shape).to(dt)


def row_module(m, runner: Runner, cov: Coverage, name: str):
    """The DeviceRows for module m, or None when its kind is not taken (an activation the
    target cannot run exactly: GELU's erf form needs the table functions)."""
    import torch.nn as nn
    cls = type(m).__name__
    if isinstance(m, nn.LayerNorm) and len(m.normalized_shape) == 1 and \
            m.weight is not None and m.bias is not None:
        return DeviceRows(runner, cov, name, "layer_norm", w=m.weight, b=m.bias, eps=m.eps)
    if cls.endswith("RMSNorm") and cls in RMS_NORMS and getattr(m, "weight", None) is not None:
        eps = getattr(m, "variance_epsilon", getattr(m, "eps", None))
        if eps is not None:
            return DeviceRows(runner, cov, name, "rms_norm", w=m.weight, eps=eps)
    fn = ACT_MODULES.get(cls)
    if cls in ("GELU", "GELUActivation") and getattr(m, "approximate", "none") == "tanh":
        fn = "gelu_tanh"
    if fn == "gelu" and not getattr(runner.cfg, "TFN", False):
        return None
    if fn is not None:
        return DeviceRows(runner, cov, name, "act", fn=fn)
    return None


# RMSNorms whose forward is x * rsqrt(mean(x^2) + eps) * weight (not Gemma's 1 + weight)
RMS_NORMS = {"LlamaRMSNorm", "Qwen2RMSNorm", "Qwen3RMSNorm", "MistralRMSNorm", "Idefics3RMSNorm",
             "SmolLM3RMSNorm"}
ACT_MODULES = {"SiLU": "silu", "SiLUActivation": "silu", "Sigmoid": "sigmoid", "Tanh": "tanh",
               "ReLU": "relu", "GELUTanh": "gelu_tanh", "PytorchGELUTanh": "gelu_tanh",
               "GELU": "gelu", "GELUActivation": "gelu"}


def device_ops(runner: Runner, cov: Coverage, min_elems: int = 1024):
    """A TorchFunctionMode running the sums, differences and products of two fp32 tensors (the
    same shape, or the second a [N] vector over the last dimension) on the device
    (kernels/eltwise.py binary), from min_elems outputs up; anything else is PyTorch's."""
    import torch
    from torch.overrides import TorchFunctionMode
    from .kernels import eltwise as E
    ops = {"add": "add", "__add__": "add", "__radd__": "add", "mul": "mul", "__mul__": "mul",
           "__rmul__": "mul", "sub": "sub", "__sub__": "sub"}

    class Ops(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            op = ops.get(getattr(func, "__name__", ""))
            if op is not None and len(args) == 2 and not kwargs:
                a, b = args
                if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and \
                        a.dtype == b.dtype == torch.float32 and a.dim() >= 1 and \
                        a.numel() >= min_elems and not (a.requires_grad or b.requires_grad):
                    N = a.shape[-1]
                    if b.shape == a.shape:
                        y = runner.rows(E.binary, {"a": a.reshape(-1, N).numpy(),
                                                   "b": b.reshape(-1, N).numpy()}, N, op=op)
                    elif b.numel() == N and b.shape[-1] == N and b.dim() <= a.dim():
                        y = runner.rows(E.binary, {"a": a.reshape(-1, N).numpy()}, N,
                                        {"b": b.reshape(N).numpy()}, op=op, brow=True)
                    else:
                        return func(*args, **kwargs)
                    cov.delems[op] += y.size
                    return torch.from_numpy(y).reshape(a.shape)
            return func(*args, **kwargs)
    return Ops()


def to_device(model, runner: Runner, cov: Coverage, select=None, wsplit: bool = False,
              convs: bool = True, rows: bool = False) -> int:
    """Every nn.Linear (and, with convs, every nn.Conv2d the kernel takes) of `model`
    (select(name, module) -> bool, default all) runs on the device; with rows, also its
    activations, LayerNorms and RMSNorms (row_module). Returns the number of modules moved."""
    import torch.nn as nn
    n = 0
    for name, m in model.named_modules():
        if select is not None and not select(name, m):
            continue
        if isinstance(m, nn.Linear):
            m.forward = DeviceLinear(m, runner, cov, name, wsplit)
            n += 1
        elif convs and isinstance(m, nn.Conv2d):
            m.forward = DeviceConv2d(m, runner, cov, name, wsplit)
            n += 1
        elif rows:
            r = row_module(m, runner, cov, name)
            if r is not None:
                m.forward = r
                n += 1
    return n
