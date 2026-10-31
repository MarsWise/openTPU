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

--fpga-nms: implies --strict-fpga, then computes class selection, box corners and
class-aware NMS on a VCMP device and reads only fixed-size final detections. This
path skips the CPU model/reference calculation. Input letterbox/resize still
runs on the host; the complete path is not yet validated with a real r6 card.

--fpga-u8: after host image decoding and letterbox resize, transfer packed uint8
pixels and convert and place them in the model input buffer on the VU8 device.

--fpga-resize: implies --fpga-u8 and moves the letterbox too: the host only decodes the image
file and writes its raw RGB bytes at their original size; the device converts them to fp32,
resizes them (kernels/letterbox.py: Pillow's bilinear filters, vertical pass on the VPU,
horizontal one and the BGR order on the MXU), pads with 114 and writes the model's input.
The float reference keeps Pillow's letterbox, so the comparison includes the resize's
difference (about a level, tests/test_letterbox.py).
"""
import argparse
import hashlib
import sys
import time

import numpy as np


def load_image_u8(path, size):
    """The existing letterbox operation, retaining byte pixels instead of making fp32."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    r = min(size / im.height, size / im.width)
    im = im.resize((int(im.width * r), int(im.height * r)), Image.BILINEAR)
    out = np.full((size, size, 3), 114, np.uint8)
    a = np.asarray(im, np.uint8)[:, :, ::-1]
    out[:a.shape[0], :a.shape[1]] = a
    return out


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
    ap.add_argument("--strict-fpga-only", action="store_true",
                    help="reject execution until image preprocessing and all inference run on the FPGA")
    ap.add_argument("--fpga-nms", action="store_true",
                    help="run decoded YOLOX class selection and NMS on r6; report only final detections")
    ap.add_argument("--fpga-u8", action="store_true",
                    help="transfer uint8 pixels and convert them to fp32 on VU8 firmware")
    ap.add_argument("--fpga-resize", action="store_true",
                    help="letterbox resize on the device too (implies --fpga-u8)")
    ap.add_argument("--max-detections", type=int, default=100)
    ap.add_argument("--confidence", type=float, default=0.3)
    ap.add_argument("--nms-iou", type=float, default=0.45)
    ap.add_argument("--wsplit", action="store_true", help="int8 hi + lo convolution weights")
    a = ap.parse_args()
    if a.fpga_resize:
        a.fpga_u8 = True
    if a.strict_fpga_only and not (a.fpga_resize and a.fpga_nms):
        ap.error("strict FPGA-only execution needs --fpga-resize --fpga-nms (the host then only "
                 "decodes the image file and reads the final detections)")
    if a.fpga_u8 and not a.image:
        ap.error("--fpga-u8 requires an explicit image")
    if a.fpga_nms:
        a.strict_fpga = True
        if a.per_op or a.check_isa:
            ap.error("--fpga-nms currently requires --per-op and --check-isa off")
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
                     DRAM_BYTES=1 << 28, VCMP=a.fpga_nms, VU8=a.fpga_u8)
    if a.fpga_nms and not cfg.VCMP:
        raise SystemExit("--fpga-nms requires r6 firmware with CAPS2 VCMP")
    if a.fpga_u8 and not cfg.VU8:
        raise SystemExit("--fpga-u8 requires r6 firmware with CAPS2 VU8")
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
        dec["n"], dec["cols"] = total, cols
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
    nms_prog = None
    image_size = plan.size
    if a.fpga_nms:
        from opentpu.kernels import nms_fpga
        n = dec["n"]
        planes = -(-plan.size // cfg.D) * cfg.D
        result = -(-(planes + 24 * n) // cfg.D) * cfg.D
        image_size = -(-(result + 28 * a.max_detections) // cfg.D) * cfg.D
        if image_size > cfg.DRAM_BYTES:
            raise SystemExit("--fpga-nms buffers exceed the configured DRAM size")
        nms_prog = nms_fpga.yolox_preprocess(n, dec["cols"], dec["val"].buf.addr, planes,
                                             tmem_words=cfg.TMEM_WORDS)
        nms_prog += nms_fpga.program(n, a.max_detections, planes, result,
                                     tmem_words=cfg.TMEM_WORDS, imem_words=cfg.IMEM_WORDS,
                                     score_threshold=a.confidence, iou_threshold=a.nms_iou)
        if len(nms_prog) * 8 > cfg.IMEM_WORDS:
            raise SystemExit("--fpga-nms combined program does not fit instruction memory")
    input_prog = None
    raw_at = None
    if a.fpga_u8:
        from opentpu.kernels.u8_input import program_rows
        b = plan.inputs["x"]
        raw_count = b.H * b.W * b.C
        raw_at = -(-image_size // cfg.D) * cfg.D
        scratch = -(-(raw_at + raw_count) // cfg.D) * cfg.D
        image_size = -(-(scratch + 4 * raw_count) // cfg.D) * cfg.D
        if image_size > cfg.DRAM_BYTES:
            raise SystemExit("--fpga-u8 buffers exceed the configured DRAM size")
        dst = b.addr + 4 * (b.border * b.rs + b.border * b.C)
        if a.fpga_resize:
            from PIL import Image
            from opentpu.kernels import letterbox as LB
            from opentpu.kernels.u8_input import program as u8_program
            raw0 = np.asarray(Image.open(a.image).convert("RGB"), np.uint8)
            H0, W0 = raw0.shape[:2]
            lb = LB.Plan(H0, W0, a.size)
            lq, lsc = lb.weights(cfg.D)
            raw_at = -(-image_size // cfg.D) * cfg.D
            scratch = -(-(raw_at + raw0.size + 4) // cfg.D) * cfg.D
            lw = -(-(scratch + 4 * raw0.size) // 4096) * 4096
            ls = -(-(lw + lq.nbytes) // 4096) * 4096
            lbias = -(-(ls + lsc.nbytes) // 4096) * 4096
            image_size = -(-(lbias + lb.bias.nbytes) // cfg.D) * cfg.D
            if image_size > cfg.DRAM_BYTES:
                raise SystemExit("--fpga-resize buffers exceed the configured DRAM size")
            input_prog = [u8_program(cfg, raw_at, scratch, raw0.size)] + \
                LB.programs(cfg, lb, scratch, lw, ls, lbias, dst, b.rs)
        else:
            input_prog = [program_rows(cfg, raw_at, dst, scratch, b.H, b.W * b.C, b.rs)]
    img = plan.image(image_size, cfg.D)
    if a.fpga_resize:
        for at, arr in ((raw_at, raw0), (lw, lq), (ls, lsc), (lbias, lb.bias)):
            v = np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
            img[at:at + v.size] = v
        x = load_image(a.image, a.size) if (a.per_op or not a.fpga_nms) else None
    elif a.fpga_u8:
        raw = load_image_u8(a.image, a.size)
        img[raw_at:raw_at + raw.size] = raw.reshape(-1)
        x = raw.transpose(2, 0, 1).astype(np.float32) if (a.per_op or not a.fpga_nms) else None
    else:
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
        if input_prog is not None:
            import dataclasses
            from opentpu.isasim import Machine
            for p in input_prog:
                sim = Machine(dataclasses.replace(cfg, DRAM_BYTES=len(img)), [p], [img]).run()
                img = sim.slices[0].dram
                stats["instructions"] += sim.slices[0].icount
        def each(i, op, im, icount):
            stats["instructions"] += icount
        out = plan.run_isa(cfg, img, each)
    else:
        out = run_board(board, plan, cfg, img, stats, a.check_isa, read_final=not a.fpga_nms,
                        input_prog=input_prog)
    if a.fpga_nms:
        if board is None:
            import dataclasses
            from opentpu.isasim import Machine
            sim = Machine(dataclasses.replace(cfg, DRAM_BYTES=len(out)), [nms_prog], [out]).run()
            values = sim.slices[0].dram[result:result + 28 * a.max_detections].view(np.float32)
        else:
            at = -(-(len(img) + PROG_AT_GAP) // 4096) * 4096
            board.load(at, nms_prog)
            st = board.run(timeout=60.0)
            stats["instructions"] += st["instructions"][0]
            stats["cycles"] += st["cycles"]
            values = board.read(result, 28 * a.max_detections).view(np.float32)
        rows = values.reshape(-1, 7)
        rows = rows[rows[:, 6] != 0]
        print(f"FPGA NMS: {len(rows)} valid detections (max {a.max_detections})")
        for x0, y0, x1, y1, score, cls, _ in rows:
            print(f"  class {int(cls):2d} score {score:.3f} box [{x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}]")
        print("input preprocessing: " + ("host image decode only; uint8 conversion, letterbox "
                                         "resize, " if a.fpga_resize else
                                         "host letterbox/resize; " +
                                         ("uint8 conversion, " if a.fpga_u8 else "")) +
              "graph, class selection and NMS: device")
        return 0
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
    khz = stats.get("core_khz") or 133333    # the card's CORE_KHZ (r5-m8 137.5 MHz)
    print(f"\nrun {t_run:.1f}s: {stats['instructions']} instructions"
          + (f", {stats['cycles']} card cycles = {stats['cycles'] / khz:.1f} ms at {khz / 1e3:.2f} MHz"
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


def run_board(board, plan, cfg, img, stats, check, read_final=True, input_prog=None):
    """Write the plan's image once; per op load its program above the image and run it."""
    import dataclasses
    from opentpu.isasim import Machine
    board.write(0, img)
    stats["core_khz"] = board.info().get("core_khz")
    at = -(-(len(img) + PROG_AT_GAP) // 4096) * 4096
    ref = img.copy() if check else None
    c = dataclasses.replace(cfg, DRAM_BYTES=len(img))
    for p in input_prog or ():
        board.load(at, p)
        st = board.run(timeout=60.0)
        stats["instructions"] += st["instructions"][0]
        stats["cycles"] += st["cycles"]
        if check:
            ref = Machine(c, [p], [ref]).run().slices[0].dram
    if input_prog is not None and check:
        b = plan.inputs["x"]
        got = board.read(b.addr, b.nbytes)
        if not np.array_equal(got, ref[b.addr:b.addr + b.nbytes]):
            raise SystemExit("FPGA input preprocessing (uint8 / letterbox) differs from ISA")
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
    return board_image(board, plan, None, len(img)) if read_final else None


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
