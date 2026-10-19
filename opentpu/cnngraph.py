"""A CNN on the device, its whole graph resident in the card's DRAM (kernels/conv.py ops).

`lower(gm)` takes a torch.fx GraphModule of a CNN built from YOLOX-style blocks -- a "BaseConv"
(any module with `.conv` (nn.Conv2d), `.bn` (nn.BatchNorm2d) and `.act` (SiLU / ReLU / Identity)),
a "Focus" (space-to-depth, then a BaseConv), plain nn.Conv2d, nn.MaxPool2d, nn.Upsample (x2,
nearest), torch.cat over channels, `+` and `.sigmoid()` -- and plans it as a list of device ops
over NHWC buffers in one DRAM image:

* every tensor is a buffer [H + 2b, W + 2b, C] (+ a row tail) whose border b is what its
  consumers read past the edge: zeros for convolutions, -3e38 for max pooling (NEG); a
  convolution reads its input in place, and an operator writes only the interior;
* a tensor consumed only by a concatenation is written by its producer straight into the
  concatenation's channel slice (pixels Ctot words apart): no copy, no host; one that is also
  read elsewhere is copied in (op "copy");
* `a + b` with `a` a convolution's output read only by the add becomes that convolution's
  residual (added after its activation, in its epilogue); `x.sigmoid()` of a convolution read
  only by it becomes its activation;
* Focus + its 3x3 convolution become one 6x6, stride-2 convolution of the image (an exact
  rewrite: the same products and sums, the weights rearranged).

`Plan.compile(cfg)` gives one program per op at fixed addresses, `Plan.image()` the DRAM image
(weights, biases, borders), and `run_isa` / `run_board` execute them in order, the tensors
staying in DRAM between ops: the host only launches programs. The graph's outputs (fx output
node arguments that are device tensors) are read back as NHWC arrays; what the graph does with
them after the last device op (YOLOX's flatten / permute / cat) is the caller's.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field

import numpy as np

from . import quant as Q
from .compiler import Affine, QTensor, Tensor
from .kernels import conv as C

ALIGN = 4096


@dataclass(eq=False)
class Buf:
    name: str
    H: int
    W: int
    C: int
    fill: float = 0.0             # the border's value: 0 (convolutions) or NEG (max pooling)
    border: int = 0
    tail: int = 0                 # words after each row's right border (conv windows' over-read)
    addr: int = -1                # bytes
    fixed_fill: bool = False

    @property
    def rs(self) -> int:          # row stride, words
        return (self.W + 2 * self.border) * self.C + self.tail

    @property
    def nbytes(self) -> int:
        return 4 * self.rs * (self.H + 2 * self.border)

    def interior(self, c0: int = 0, c: int | None = None) -> Tensor:
        """[H, W, c] view of channels c0 .. c0 + c (pixels C words apart)."""
        c = self.C - c0 if c is None else c
        base = self.addr + 4 * (self.border * self.rs + self.border * self.C + c0)
        return Tensor(Affine(base), (self.H, self.W, c), (self.rs, self.C, 1))

    def rows(self, pad: int, words: int) -> Tensor:
        """The rows of the input padded by `pad` (<= border) as [H + 2 pad, words]."""
        o = self.border - pad
        base = self.addr + 4 * (o * self.rs + o * self.C)
        if o * self.C + words > self.rs or pad > self.border:
            raise ValueError(f"{self.name}: a {words}-word row at pad {pad} exceeds the buffer")
        return Tensor(Affine(base), (self.H + 2 * pad, words), (self.rs, 1))


@dataclass(eq=False)
class Val:
    """A tensor value: channels c0 .. c0 + C of a buffer."""
    buf: Buf
    c0: int = 0
    C: int = -1

    def __post_init__(self):
        if self.C < 0:
            self.C = self.buf.C

    @property
    def whole(self) -> bool:
        return self.c0 == 0 and self.C == self.buf.C


@dataclass(eq=False)
class Op:
    kind: str                     # conv, dw, pool, up, copy
    name: str
    inp: Val
    out: Val
    p: dict = field(default_factory=dict)
    res: Val | None = None
    prog: list | None = None


NEG = C.NEG


def _act_name(mod) -> str:
    import torch.nn as nn
    a = getattr(mod, "act", None)
    if isinstance(a, nn.SiLU):
        return "silu"
    if isinstance(a, nn.ReLU):
        return "relu"
    if a is None or isinstance(a, nn.Identity):
        return "none"
    raise NotImplementedError(f"activation {type(a).__name__}")


def _fold(conv, bn):
    """Conv2d (+ BatchNorm2d) -> float32 weights OIHW, bias."""
    w = conv.weight.detach().double().numpy()
    b = conv.bias.detach().double().numpy() if conv.bias is not None else np.zeros(w.shape[0])
    if bn is not None:
        s = bn.weight.detach().double().numpy() / np.sqrt(bn.running_var.detach().double().numpy()
                                                           + bn.eps)
        w = w * s[:, None, None, None]
        b = (b - bn.running_mean.detach().double().numpy()) * s + bn.bias.detach().double().numpy()
    return w.astype(np.float32), b.astype(np.float32)


def focus_weights(w3: np.ndarray) -> np.ndarray:
    """Focus (cat of x[::2, ::2], x[1::2, ::2], x[::2, 1::2], x[1::2, 1::2] over channels) then a
    3x3 / stride 1 / pad 1 convolution, as one 6x6 / stride 2 / pad 2 convolution of the image:
    tap (r, s) of input channel c is the 3x3 weight of channel q*Cin + c at (r // 2, s // 2),
    q the slice (dy, dx) = (r % 2, s % 2) in Focus's order."""
    Cout, C12, kh, kw = w3.shape
    if (kh, kw) != (3, 3) or C12 % 4:
        raise NotImplementedError("Focus followed by a 3x3 convolution of 4 * Cin channels")
    cin = C12 // 4
    q_of = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
    w6 = np.zeros((Cout, cin, 6, 6), np.float32)
    for r in range(6):
        for s in range(6):
            q = q_of[(r % 2, s % 2)]
            w6[:, :, r, s] = w3[:, q * cin:(q + 1) * cin, r // 2, s // 2]
    return w6


class Plan:
    def __init__(self, wsplit: bool = False):
        self.wsplit = wsplit                    # convolutions' weights as int8 hi + lo (split_weight)
        self.ops: list[Op] = []
        self.bufs: list[Buf] = []
        self.inputs: dict[str, Buf] = {}
        self.outputs: list[Val] = []
        self.weights: dict[str, tuple] = {}     # op name -> (packed fp32 [N, K] or taps, bias)
        self.consts: list[tuple] = []           # (name, fp32 array, address) placed with layout
        self.extra: list[tuple] = []            # (Op, kernel, bound(plan) -> dict): appended ops

    def const(self, name: str, a: np.ndarray) -> list:
        """A constant fp32 array placed in the image; returns a holder whose [2] is its address
        once the plan is laid out."""
        h = [name, np.ascontiguousarray(a, np.float32), -1]
        self.consts.append(h)
        return h

    def kernel_op(self, name: str, kernel, bound, out: "Val | None" = None) -> Op:
        """A device op of any kernel: bound(plan) gives its arguments once addresses are known."""
        op = Op("kernel", name, None, out)
        self.extra.append((op, kernel, bound))
        self.ops.append(op)
        return op

    def buf(self, name, H, W, Cc) -> Buf:
        b = Buf(name, H, W, Cc)
        self.bufs.append(b)
        return b

    # ------------------------------------------------------------------ layout
    def layout(self, D: int) -> int:
        """Borders and tails from the consumers, then addresses. Returns the image size."""
        need = {}
        for op in self.ops:
            if op.inp is None:
                continue
            b = op.inp.buf
            if op.kind == "conv":
                K, s, pad = op.p["K"], op.p["stride"], op.p["pad"]
                rw = C.geometry(b.H, b.W, op.inp.C, K, K, s, pad, D)[5]   # Kp-wide windows' row
                need.setdefault(id(b), []).append(("zero", pad, rw))
            elif op.kind == "dw":
                need.setdefault(id(b), []).append(("zero", op.p["pad"], (b.W + 2 * op.p["pad"]) * b.C))
            elif op.kind == "pool":
                need.setdefault(id(b), []).append(("neg", op.p["pad"], (b.W + 2 * op.p["pad"]) * b.C))
        for b in self.bufs:
            reqs = need.get(id(b), [])
            fills = {r[0] for r in reqs if r[1] > 0}
            if len(fills) > 1:
                raise NotImplementedError(f"{b.name}: read with both zero and -inf padding")
            b.fill = NEG if fills == {"neg"} else 0.0
            b.border = max([r[1] for r in reqs], default=0)
            b.tail = 0
            for _, pad, words in reqs:
                b.tail = max(b.tail, (b.border - pad) * b.C + words - (b.W + 2 * b.border) * b.C)
        top = 0
        for b in self.bufs:
            b.addr = top
            top = -(-(top + b.nbytes) // ALIGN) * ALIGN
        for h in self.consts:
            h[2] = top
            top = -(-(top + h[1].nbytes) // ALIGN) * ALIGN
        for op in self.ops:
            if op.kind == "conv":
                w2, bias = self.weights[op.name]
                n, k = w2.shape
                for t in (("w_at", "s_at"), ("wl_at", "sl_at")) if self.wsplit else (("w_at", "s_at"),):
                    op.p[t[0]] = top
                    top = -(-(top + n * Q.row_bytes(k, "int8", D)) // ALIGN) * ALIGN
                    op.p[t[1]] = top
                    top = -(-(top + n * Q.scale_bytes(k, "int8", D)) // ALIGN) * ALIGN
            elif op.kind == "dw":
                taps, bias = self.weights[op.name]
                op.p["w_at"] = top
                top = -(-(top + taps.nbytes) // ALIGN) * ALIGN
            if op.kind in ("conv", "dw"):
                op.p["b_at"] = top
                top = -(-(top + 4 * self.weights[op.name][1].size) // ALIGN) * ALIGN
        return top

    def image(self, size: int, D: int) -> np.ndarray:
        img = np.zeros(size, np.uint8)
        for _, a, at in self.consts:
            img[at:at + a.nbytes] = a.view(np.uint8).reshape(-1)
        f32 = img[:size - size % 4].view(np.float32)
        for b in self.bufs:
            if b.fill:
                f32[b.addr // 4:(b.addr + b.nbytes) // 4] = b.fill
        for op in self.ops:
            if op.kind == "conv":
                w2, bias = self.weights[op.name]
                mats = C.split_weight(w2, D) if self.wsplit else (w2,)
                for m, (wa, sa) in zip(mats, (("w_at", "s_at"), ("wl_at", "sl_at"))):
                    q, sc = Q.quantize_mxu(m, "int8", D)
                    img[op.p[wa]:op.p[wa] + q.nbytes] = q.view(np.uint8).reshape(-1)
                    img[op.p[sa]:op.p[sa] + sc.nbytes] = sc.view(np.uint8).reshape(-1)
            elif op.kind == "dw":
                taps, bias = self.weights[op.name]
                img[op.p["w_at"]:op.p["w_at"] + taps.nbytes] = taps.view(np.uint8).reshape(-1)
            if op.kind in ("conv", "dw"):
                bias = self.weights[op.name][1]
                img[op.p["b_at"]:op.p["b_at"] + bias.nbytes] = bias.view(np.uint8)
        return img

    def put_input(self, img: np.ndarray, name: str, x: np.ndarray) -> None:
        """An NHWC input array into its buffer's interior."""
        b = self.inputs[name]
        f32 = img.view(np.float32)
        for h in range(b.H):
            o = (b.addr + 4 * ((b.border + h) * b.rs + b.border * b.C)) // 4
            f32[o:o + b.W * b.C] = x[h].reshape(-1)

    def read(self, img: np.ndarray, v: Val) -> np.ndarray:
        """A value's NHWC array from a DRAM image."""
        b = v.buf
        f32 = img.view(np.float32)
        out = np.empty((b.H, b.W, v.C), np.float32)
        for h in range(b.H):
            o = (b.addr + 4 * ((b.border + h) * b.rs + b.border * b.C)) // 4
            out[h] = f32[o:o + b.W * b.C].reshape(b.W, b.C)[:, v.c0:v.c0 + v.C]
        return out

    # ------------------------------------------------------------------ compile
    def compile(self, cfg) -> None:
        D = cfg.D
        extra = {id(op): (k, bnd) for op, k, bnd in self.extra}
        for op in self.ops:
            if op.kind == "kernel":
                k, bnd = extra[id(op)]
                op.prog = k.trace(cfg, 0, bnd(self)).finish()
                continue
            ib, ov = op.inp.buf, op.out
            y = ov.buf.interior(ov.c0, ov.C)
            res = op.res.buf.interior(op.res.c0, op.res.C) if op.res is not None else None
            if op.kind == "conv":
                K, s, pad = op.p["K"], op.p["stride"], op.p["pad"]
                OH, OW, Hp, Wp, Kp, rw = C.geometry(ib.H, ib.W, ib.C, K, K, s, pad, D)
                w2 = self.weights[op.name][0]
                bound = dict(x=ib.rows(pad, rw),
                             w=QTensor.weight(op.p["w_at"], op.p["s_at"], w2.shape[0], w2.shape[1],
                                              "int8", D),
                             b=Tensor(Affine(op.p["b_at"]), (ov.C,), (1,)), y=y, H=ib.H, W=ib.W,
                             Cin=ib.C, Cout=ov.C, KH=K, KW=K, stride=s, pad=pad, act=op.p["act"],
                             res=res,
                             wlo=(QTensor.weight(op.p["wl_at"], op.p["sl_at"], w2.shape[0],
                                                 w2.shape[1], "int8", D) if self.wsplit else None))
                kern = C.conv2d
            elif op.kind == "dw":
                K, s, pad = op.p["K"], op.p["stride"], op.p["pad"]
                bound = dict(x=ib.rows(pad, (ib.W + 2 * pad) * ib.C),
                             w=Tensor(Affine(op.p["w_at"]), (K * K, ib.C), (ib.C, 1)),
                             b=Tensor(Affine(op.p["b_at"]), (ib.C,), (1,)), y=y, H=ib.H, W=ib.W,
                             C=ib.C, K=K, stride=s, pad=pad, act=op.p["act"], res=res)
                kern = C.depthwise2d
            elif op.kind == "pool":
                K, pad = op.p["K"], op.p["pad"]
                bound = dict(x=ib.rows(pad, (ib.W + 2 * pad) * ib.C), y=y, H=ib.H, W=ib.W, C=ib.C,
                             K=K, stride=1, pad=pad)
                kern = C.maxpool2d
            elif op.kind == "up":
                bound = dict(x=ib.rows(0, ib.W * ib.C), y=y, H=ib.H, W=ib.W, C=ib.C)
                kern = C.upsample2x
            elif op.kind == "copy":
                bound = dict(x=ib.interior(), y=y, H=ib.H, W=ib.W, C=ib.C)
                kern = C.copy_into
            else:
                raise NotImplementedError(op.kind)
            op.prog = kern.trace(cfg, 0, bound).finish()

    # ------------------------------------------------------------------ run
    def run_isa(self, cfg, img: np.ndarray, each=None) -> np.ndarray:
        from .isasim import Machine
        import dataclasses
        c = dataclasses.replace(cfg, DRAM_BYTES=len(img))
        for i, op in enumerate(self.ops):
            sl = Machine(c, [op.prog], [img]).run().slices[0]
            img = sl.dram
            if each is not None:
                each(i, op, img, sl.icount)
        return img


def lower(gm, input_shape, wsplit: bool = False) -> Plan:
    """Plan a traced CNN (see the module docstring). input_shape: (C, H, W) of the one input."""
    import torch
    import torch.nn as nn
    pl = Plan(wsplit)
    env = {}
    users = {n: list(n.users) for n in gm.graph.nodes}
    mods = dict(gm.named_modules())

    def is_baseconv(m):
        return hasattr(m, "conv") and hasattr(m, "bn") and isinstance(m.conv, nn.Conv2d)

    def shape_of(v: Val):
        return v.buf.H, v.buf.W, v.C

    def new_out(node, H, W, Cc) -> Val:
        return Val(pl.buf(node.name, H, W, Cc))

    def conv_op(node, conv, bn, act, x: Val, name, w=None, stride=None, pad=None):
        wf, b = _fold(conv, bn)
        if w is not None:
            wf = w
        H, W, Cin = shape_of(x)
        K = wf.shape[-1]
        s = conv.stride[0] if stride is None else stride
        p = conv.padding[0] if pad is None else pad
        OH, OW = (H + 2 * p - K) // s + 1, (W + 2 * p - K) // s + 1
        g = conv.groups
        out = new_out(node, OH, OW, wf.shape[0])
        if g == 1:
            op = Op("conv", name, x, out, dict(K=K, stride=s, pad=p, act=act))
            pl.weights[name] = (C.pack_conv_weight(wf, D_HINT[0]), b)
        elif g == Cin == wf.shape[0]:
            op = Op("dw", name, x, out, dict(K=K, stride=s, pad=p, act=act))
            pl.weights[name] = (C.pack_dw_weight(wf), b)
        else:
            raise NotImplementedError(f"{name}: groups {g}")
        pl.ops.append(op)
        return out, op

    producer = {}                 # id(Val) -> Op
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if not users[node]:
                continue
            Cc, H, W = input_shape
            b = pl.buf(node.name, H, W, Cc)
            pl.inputs[node.name] = b
            env[node] = Val(b)
        elif node.op == "call_module":
            m = mods[node.target]
            x = env[node.args[0]]
            if type(m).__name__ == "Focus" and is_baseconv(getattr(m, "conv", None)):
                bc = m.conv
                w3, _ = _fold(bc.conv, bc.bn)
                v, op = conv_op(node, bc.conv, bc.bn, _act_name(bc), x, node.target,
                                w=focus_weights(w3), stride=2, pad=2)
            elif is_baseconv(m):
                v, op = conv_op(node, m.conv, m.bn, _act_name(m), x, node.target)
            elif isinstance(m, nn.Conv2d):
                v, op = conv_op(node, m, None, "none", x, node.target)
            elif isinstance(m, nn.MaxPool2d):
                if m.stride not in (1, (1, 1)) or 2 * m.padding + 1 != m.kernel_size:
                    raise NotImplementedError("max pooling other than stride 1, same size")
                H, W, Cc = shape_of(x)
                v = new_out(node, H, W, Cc)
                op = Op("pool", node.target, x, v, dict(K=m.kernel_size, pad=m.padding))
                pl.ops.append(op)
            elif isinstance(m, nn.Upsample):
                if m.scale_factor not in (2, 2.0) or m.mode != "nearest":
                    raise NotImplementedError(f"upsample {m.scale_factor} {m.mode}")
                H, W, Cc = shape_of(x)
                v = new_out(node, 2 * H, 2 * W, Cc)
                op = Op("up", node.target, x, v)
                pl.ops.append(op)
            else:
                raise NotImplementedError(f"module {type(m).__name__} ({node.target})")
            if not x.whole:
                raise NotImplementedError(f"{node.target}: input is a channel slice")
            env[node] = v
            producer[id(v)] = op
        elif node.op == "call_method" and node.target == "sigmoid":
            src = node.args[0]
            v = env[src]
            op = producer.get(id(v))
            if op is None or op.kind != "conv" or op.p["act"] != "none" or len(users[src]) != 1:
                raise NotImplementedError("sigmoid of anything but a linear convolution")
            op.p["act"] = "sigmoid"
            env[node] = v
        elif node.op == "call_function" and node.target in (operator.add, torch.add):
            a, b = node.args[:2]
            va, vb = env[a], env[b]
            op = producer.get(id(va))
            if not (op and op.kind in ("conv", "dw") and op.res is None and len(users[a]) == 1):
                op, va, vb, a = producer.get(id(vb)), vb, va, b
            if not (op and op.kind in ("conv", "dw") and op.res is None and len(users[a]) == 1):
                raise NotImplementedError("an add that is not a convolution's residual")
            if shape_of(va) != shape_of(vb):
                raise NotImplementedError("residual of another shape")
            op.res = vb
            env[node] = va
        elif node.op == "call_function" and node.target is torch.cat:
            parts = node.args[0]
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
            if any(env.get(p) is None for p in parts):
                env[node] = None              # of host values (YOLOX's flattened levels)
                continue
            if dim != 1:
                raise NotImplementedError(f"cat over dim {dim}")
            vals = [env[p] for p in parts]
            H, W, _ = shape_of(vals[0])
            if any(shape_of(v)[:2] != (H, W) for v in vals):
                raise NotImplementedError("cat of different sizes")
            total = sum(v.C for v in vals)
            cb = pl.buf(node.name, H, W, total)
            c0 = 0
            for p, v in zip(parts, vals):
                op = producer.get(id(v))
                if op is not None and len(users[p]) == 1 and v.whole and v.buf is not cb:
                    # the producer writes straight into the slice; its own buffer goes unused
                    old = v.buf
                    v.buf, v.c0, v.C = cb, c0, v.C
                    pl.bufs.remove(old)
                else:
                    sl = Val(cb, c0, v.C)
                    cp = Op("copy", f"{node.name}.copy{c0}", v, sl)
                    pl.ops.append(cp)
                c0 += v.C
            env[node] = Val(cb)
            producer[id(env[node])] = None
        elif node.op == "output":
            out = node.args[0]
            pl.out_node = out
        else:
            env[node] = None              # shape queries, flatten, permute: the caller's
    pl.env = env
    return pl


D_HINT = [128]                            # pack_conv_weight's D: set by build() before lower()


HOST_LAYOUT = {"flatten", "permute", "getattr", "getitem", "cat", "view", "reshape", "size",
               "contiguous", "type"}


def host_nodes(gm, plan) -> list:
    """The graph's nodes the plan leaves to the host (env None), as (name, op, target)."""
    out = []
    for n in gm.graph.nodes:
        if n.op in ("placeholder", "output") or isinstance(plan.env.get(n), Val):
            continue
        if n.op == "placeholder":
            continue
        t = getattr(n.target, "__name__", str(n.target))
        out.append((n.name, n.op, t))
    return out


def build(gm, input_shape, cfg, extend=None, wsplit: bool = False) -> Plan:
    """lower, then extend(plan) (appended device ops, e.g. a detection decode), layout, compile.
    wsplit: convolution weights as int8 hi + lo (two MMs a kernel row, ~16-bit weights)."""
    D_HINT[0] = cfg.D
    pl = lower(gm, input_shape, wsplit)
    if extend is not None:
        extend(pl)
    size = pl.layout(cfg.D)
    pl.size = size
    pl.compile(cfg)
    return pl
