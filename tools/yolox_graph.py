"""YOLOX-Nano with its whole graph on the device (opentpu/cnngraph.py): every tensor stays in the
card's DRAM between ops, the host only launches one program per op and, at the end, reads the
three head levels, flattens them and decodes the boxes (grid, stride, exp) and runs NMS.

Usage: python tools/yolox_graph.py YOLOX_REPO yolox_nano.pth [--image IMG] [--size 416]
                                   [--backend isa|board] [--check-isa] [--dev /dev/xdma0]
                                   [--stubs DIR] [--strict-fpga]
Compared with the float PyTorch model (the same checkpoint and input): the head outputs, every
op's output tensor (--per-op), and the detections. --backend board writes the plan's DRAM image
once, then loads and runs each op's program; --check-isa runs the ISA simulator alongside and
requires each op's output region to be bit-identical.

--strict-fpga: the box decode runs on the device too (kernels/detect.py: every level decoded
into one [anchors, 85] buffer, the levels' flatten and concatenation included), and the run fails
if the plan leaves the host anything but reading that buffer and NMS: graph nodes the plan does
not run must be pure layout (flatten / permute / the levels' cat / shape queries), which the
decode buffer's layout replaces. A placement report lists where every operator ran.
"""
import argparse
import hashlib
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("weights")
    ap.add_argument("--image", default=None)
    ap.add_argument("--size", type=int, default=416)
    ap.add_argument("--backend", default="isa", choices=("isa", "board"))
    ap.add_argument("--check-isa", action="store_true")
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--stubs", default=None)
    ap.add_argument("--per-op", action="store_true")
    ap.add_argument("--strict-fpga", action="store_true")
    ap.add_argument("--wsplit", action="store_true", help="int8 hi + lo convolution weights")
    a = ap.parse_args()
    sys.path.insert(0, a.repo)
    if a.stubs:
        sys.path.insert(0, a.stubs)
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import torch
    import torch.fx as fx
    import torch.nn as nn
    from yolox.exp import get_exp
    from yolox.models.network_blocks import BaseConv, Focus
    from opentpu import Config, cnngraph as G
    from yolox_run import detect, iou, load_image, rel

    h = hashlib.sha256(open(a.weights, "rb").read()).hexdigest()
    exp = get_exp(f"{a.repo}/exps/default/yolox_nano.py", None)
    m = exp.get_model().eval()
    m.load_state_dict(torch.load(a.weights, map_location="cpu", weights_only=False)["model"])
    print(f"YOLOX-Nano sha256 {h}, input {a.size}x{a.size}, backend {a.backend}")

    class Leaf(fx.Tracer):
        def is_leaf_module(self, mod, qn):
            return isinstance(mod, (BaseConv, Focus, nn.MaxPool2d, nn.Upsample, nn.Conv2d)) or \
                super().is_leaf_module(mod, qn)

    m.head.decode_in_inference = False
    gm = fx.GraphModule(m, Leaf().trace(m))
    board = None
    if a.backend == "board":
        import dataclasses
        from opentpu.host.board import Board, XdmaTransport, device_config
        board = Board(XdmaTransport(a.dev), check=False)
        info = board.info()
        cfg = dataclasses.replace(device_config(info, DRAM_BYTES=1 << 28), S=1)
        print(f"card build {info.get('build_id', 0):#x}: D={cfg.D} MCOLS={cfg.MCOLS} "
              f"ACT_ROWS={cfg.act_rows} ACT_BLOCKS={cfg.ACT_BLOCKS} TMEM={cfg.TMEM_WORDS}")
    else:
        cfg = Config(S=1, D=128, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16, TMEM_WORDS=65536, LANES=8,
                     DRAM_BYTES=1 << 28)
    t0 = time.time()
    dec = {}

    def extend(pl):
        """--strict-fpga: decode the head levels on the device into one [anchors, 85] buffer."""
        from opentpu.compiler import Affine, Tensor
        from opentpu.kernels import detect as K
        lv = [n for n in gm.graph.nodes if isinstance(pl.env.get(n), G.Val) and
              any(pl.env.get(u) is None for u in n.users)]
        sizes = [(pl.env[n].buf.H, pl.env[n].buf.W, pl.env[n].C) for n in lv]
        total = sum(h * w for h, w, _ in sizes)
        cols = sizes[0][2]
        out = pl.buf("decoded", total, 1, cols)
        r0 = 0
        for n, (h, w, c) in zip(lv, sizes):
            g = pl.const(f"grid{h}x{w}", K.grid(h, w))
            v = pl.env[n]

            def bound(p, v=v, g=g, h=h, w=w, r0=r0):
                return dict(x=Tensor(Affine(v.buf.addr), (h * w, cols), (cols, 1)),
                            g=Tensor(Affine(g[2]), (h * w, 2), (2, 1)),
                            y=Tensor(Affine(out.addr + 4 * r0 * cols), (h * w, cols), (cols, 1)),
                            n=h * w, cols=cols, stride=a.size // h)
            pl.kernel_op(f"decode{h}x{w}", K.yolox_decode, bound, G.Val(out))
            r0 += h * w
        dec["val"] = G.Val(out)

    plan = G.build(gm, (3, a.size, a.size), cfg, extend if a.strict_fpga else None,
                   wsplit=a.wsplit)
    if a.strict_fpga:
        bad = [t for t in G.host_nodes(gm, plan) if t[2] not in G.HOST_LAYOUT]
        if bad:
            raise SystemExit(f"--strict-fpga: operators left to the host: {bad}")
    kinds = {}
    for op in plan.ops:
        kinds[op.kind] = kinds.get(op.kind, 0) + 1
    print(f"plan: {len(plan.ops)} ops {kinds}, {len(plan.bufs)} buffers, DRAM image "
          f"{plan.size / 2**20:.1f} MiB, instructions {sum(len(op.prog) for op in plan.ops)} "
          f"(compiled in {time.time() - t0:.1f}s)")
    img = plan.image(plan.size, cfg.D)
    x = load_image(a.image, a.size)
    plan.put_input(img, "x", x.transpose(1, 2, 0))

    # the float model's per-node values (fx interpreter), for --per-op
    vals = {}
    if a.per_op:
        class Rec(fx.Interpreter):
            def run_node(self, n):
                r = super().run_node(n)
                if isinstance(r, torch.Tensor) and r.dim() == 4:
                    vals[n.name] = r[0].permute(1, 2, 0).detach().numpy()
                return r
        with torch.no_grad():
            Rec(gm).run(torch.from_numpy(x)[None])

    stats = {"instructions": 0, "cycles": 0}
    t0 = time.time()
    if board is None:
        def each(i, op, im, icount):
            stats["instructions"] += icount
        out = plan.run_isa(cfg, img, each)
    else:
        out = run_board(board, plan, cfg, img, stats, a.check_isa)
    t_run = time.time() - t0

    with torch.no_grad():
        yr_raw = m(torch.from_numpy(x)[None])[0].numpy()             # [anchors, 85], pre-decode
        m.head.decode_in_inference = True
        yr = m(torch.from_numpy(x)[None])[0].numpy()
    levels = [n for n in gm.graph.nodes if isinstance(plan.env.get(n), G.Val) and
              any(plan.env.get(u) is None for u in n.users)]
    heads = [plan.read(out, plan.env[n]) for n in levels]
    raw = np.concatenate([h.reshape(-1, h.shape[-1]) for h in heads], 0)
    if a.strict_fpga:
        yd = plan.read(out, dec["val"]).reshape(-1, raw.shape[1])
    else:
        yd = decode(raw, [h.shape[:2] for h in heads], a.size)
    print(f"\nrun {t_run:.1f}s: {stats['instructions']} instructions"
          + (f", {stats['cycles']} card cycles = {stats['cycles'] / 133.333e3:.1f} ms at 133.33 MHz"
             if stats["cycles"] else ""))
    print(f"head output [{raw.shape[0]}, {raw.shape[1]}] vs float: rel L2 {rel(raw, yr_raw):.4f}, "
          f"box (decoded) rel L2 {rel(yd[:, :4], yr[:, :4]):.4f}, objectness max|d| "
          f"{np.abs(yd[:, 4] - yr[:, 4]).max():.4f}")
    if a.per_op:
        errs, last = [], {}
        for n in gm.graph.nodes:            # a fused add / sigmoid shares its conv's value: the
            v = plan.env.get(n)             # value is compared with the last node that names it
            if isinstance(v, G.Val) and n.op != "placeholder":
                last[id(v)] = (n, v)
        for n, v in last.values():
            if n.name in vals:
                errs.append((rel(plan.read(out, v), vals[n.name]), n.name))
        errs.sort()
        print(f"per-op outputs vs float ({len(errs)}): median {np.median([e for e, _ in errs]):.4f}, "
              f"max {errs[-1][0]:.4f} ({errs[-1][1]})")
    dr, dd = detect(yr), detect(yd)
    print(f"\ndetections (score > 0.3): float {len(dr)}, device {len(dd)}")
    for tag, det in (("float", dr), ("device", dd)):
        for x0, y0, x1, y1, sc, c in det[:10]:
            print(f"  {tag:6s} class {int(c):2d} score {sc:.3f} box [{x0:.1f} {y0:.1f} {x1:.1f} {y1:.1f}]")
    match = [max((iou(p[:4], q[:4]) for q in dd if q[5] == p[5]), default=0.0) for p in dr]
    print("float detections' best same-class IoU on the device: " +
          ", ".join(f"{v:.3f}" for v in match))
    if a.strict_fpga:
        print("placement (--strict-fpga): device", {k: v for k, v in kinds.items()},
              "| host: input write, one [anchors, 85] read, NMS | layout-only graph nodes "
              f"replaced by the decode buffer: {len(G.host_nodes(gm, plan))}")
    else:
        print("host work: input write, 3 head reads, flatten, box decode (grid / exp / stride), NMS")
    return 0


def decode(raw, sizes, img_size):
    """YOLOXHead.decode_outputs in numpy: xy = (o + grid) * stride, wh = exp(o) * stride."""
    out = raw.copy()
    o = 0
    for (h, w) in sizes:
        s = img_size // h
        yv, xv = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        g = np.stack([xv, yv], -1).reshape(-1, 2)
        n = h * w
        out[o:o + n, :2] = (raw[o:o + n, :2] + g) * s
        out[o:o + n, 2:4] = np.exp(raw[o:o + n, 2:4]) * s
        o += n
    return out


PROG_AT_GAP = 1 << 20


def run_board(board, plan, cfg, img, stats, check):
    """Write the plan's image once; per op load its program above the image and run it."""
    import dataclasses
    from opentpu.isasim import Machine
    board.write(0, img)
    at = -(-(len(img) + PROG_AT_GAP) // 4096) * 4096
    ref = img.copy() if check else None
    c = dataclasses.replace(cfg, DRAM_BYTES=len(img))
    for i, op in enumerate(plan.ops):
        board.load(at, op.prog)
        st = board.run(timeout=60.0)
        stats["instructions"] += st["instructions"][0]
        stats["cycles"] += st["cycles"]
        if check:
            ref = Machine(c, [op.prog], [ref]).run().slices[0].dram
            v = op.out
            want = plan.read(ref, v)
            got = plan.read(board_image(board, plan, v), v)
            if not np.array_equal(got.view(np.uint32), want.view(np.uint32)):
                bad = int((got.view(np.uint32) != want.view(np.uint32)).sum())
                raise SystemExit(f"op {i} {op.kind} {op.name}: {bad} words differ from the ISA")
    stats["checked ops"] = len(plan.ops) if check else 0
    return board_image(board, plan, None, len(img))


def board_image(board, plan, v, size=None):
    """The card's DRAM: the buffer of v (or the first `size` bytes) in an image-sized array."""
    if v is None:
        return board.read(0, size)
    b = v.buf
    img = np.zeros(b.addr + b.nbytes, np.uint8)
    img[b.addr:b.addr + b.nbytes] = board.read(b.addr, b.nbytes)
    return img


if __name__ == "__main__":
    sys.exit(main())
