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
    image, outputs required bit-identical."""

    def __init__(self, backend: str = "isa", cfg=None, dev: str = "/dev/xdma0",
                 check_isa: bool = False):
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

    def run(self, kernel, **args) -> np.ndarray:
        from .runtime import compile_kernel, run
        comp, imgs = compile_kernel(kernel, self.cfg, **args)
        t0 = time.perf_counter()
        if self.board is None:
            y = run(comp, imgs, "isa").outputs["y"]
        else:
            y, st = run_on_board(self.board, comp, imgs[0])
            self.cycles += int((st or {}).get("cycles") or 0)
            if self.check_isa:
                z = run(comp, imgs, "isa").outputs["y"]
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
        return "\n".join(lines)


def host_counter(cov: Coverage):
    """A TorchFunctionMode counting the MACs of matmul-family ops PyTorch runs on the host
    (matmul / bmm / mm / einsum / linear / conv / scaled_dot_product_attention)."""
    import torch
    from torch.overrides import TorchFunctionMode

    def numel(t):
        return int(t.numel()) if isinstance(t, torch.Tensor) else 0

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
            return func(*args, **kwargs)
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
        y = self.runner.run(L.linear, x=Input(L.pad_rows(a, self.runner.cfg.D)), w=self.w,
                            b=Input(self.b), y=Output((M, self.N)), M=M, K=self.K, N=self.N,
                            **kw)
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


def to_device(model, runner: Runner, cov: Coverage, select=None, wsplit: bool = False,
              convs: bool = True) -> int:
    """Every nn.Linear (and, with convs, every nn.Conv2d the kernel takes) of `model`
    (select(name, module) -> bool, default all) runs on the device. Returns the number of
    layers moved."""
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
    return n
