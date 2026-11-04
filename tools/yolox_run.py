"""YOLOX-Nano on openTPU: the PyTorch model's own forward (Megvii YOLOX, Apache-2.0), with every
BaseConv (Conv2d + folded BatchNorm + SiLU), MaxPool2d and Upsample run by kernels/conv.py on the
ISA simulator; the rest of the graph runs in PyTorch on the host and is counted.

Usage: python tools/yolox_run.py YOLOX_REPO yolox_nano.pth [--image IMG] [--size 416]
                                 [--backend isa|ref|board] [--check-isa] [--dev /dev/xdma0]
YOLOX_REPO is a checkout of github.com/Megvii-BaseDetection/YOLOX (its `yolox` package and
exps/default/yolox_nano.py). --backend ref runs the kernels' numpy references instead (fast; the
same graph). --backend board runs every kernel on the card (the bitstream's Config,
opentpu.host.board.device_config); --check-isa also runs each on the ISA simulator from the same
DRAM image and requires the outputs to be bit-identical. Reports: device vs host operator counts and MACs, every BaseConv's output against
the float model's (relative L2), the head outputs, and the detections (score > 0.3, NMS 0.45)
of both.
"""
import argparse
import collections
import copy
import hashlib
import sys
import time
import types

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("weights")
    ap.add_argument("--image", default=None)
    ap.add_argument("--size", type=int, default=416)
    ap.add_argument("--backend", default="isa", choices=("isa", "ref", "board"))
    ap.add_argument("--check-isa", action="store_true")
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--stubs", default=None, help="a dir with a torchvision stub, if absent")
    a = ap.parse_args()
    sys.path.insert(0, a.repo)
    if a.stubs:
        sys.path.insert(0, a.stubs)
    import torch
    import torch.nn as nn
    from yolox.exp import get_exp
    from yolox.models.network_blocks import BaseConv
    from opentpu import Config
    from opentpu.kernels import conv as C
    from opentpu.runtime import Input, Output, Weight, compile_kernel, run

    h = hashlib.sha256(open(a.weights, "rb").read()).hexdigest()
    exp = get_exp(f"{a.repo}/exps/default/yolox_nano.py", None)
    ref = exp.get_model().eval()
    ref.load_state_dict(torch.load(a.weights, map_location="cpu", weights_only=False)["model"])
    ref.head.decode_in_inference = True
    print(f"YOLOX-Nano {a.weights} sha256 {h}, input {a.size}x{a.size}, backend {a.backend}")
    img = load_image(a.image, a.size)
    x = torch.from_numpy(img)[None]

    stats = collections.Counter()
    t_dev = [0.0]
    board = None
    if a.backend == "board":
        import dataclasses
        from opentpu.host.board import Board, XdmaTransport, device_config
        board = Board(XdmaTransport(a.dev), check=False)
        info = board.info()
        cfg = dataclasses.replace(device_config(info, DRAM_BYTES=1 << 26), S=1)
        print(f"card: build {info.get('build_id', 0):#x}, D={cfg.D} MCOLS={cfg.MCOLS} "
              f"ACT_ROWS={cfg.act_rows} ACT_BLOCKS={cfg.ACT_BLOCKS} TMEM={cfg.TMEM_WORDS}")
    else:
        cfg = Config(S=1, D=128, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16, TMEM_WORDS=65536,
                     DRAM_BYTES=1 << 26, LANES=8)

    def launch(kernel, **args):
        t0 = time.time()
        comp, imgs = compile_kernel(kernel, cfg, **args)
        if board is None:
            r = run(comp, imgs, "isa")
            stats["instructions"] += r.stats["instructions"][0]
            t_dev[0] += time.time() - t0
            return r.outputs["y"]
        y = run_on_board(board, comp, imgs[0])
        stats["instructions"] += y[1]["instructions"][0]
        stats["card cycles"] += y[1]["cycles"]
        t_dev[0] += time.time() - t0
        if a.check_isa:
            r = run(comp, [imgs[0].copy()], "isa")
            stats["launches checked bit-exact vs ISA"] += 1
            if not np.array_equal(r.outputs["y"].view(np.uint32), y[0].view(np.uint32)):
                bad = int((r.outputs["y"].view(np.uint32) != y[0].view(np.uint32)).sum())
                raise SystemExit(f"{kernel.__name__}: {bad} output words differ from the ISA")
        return y[0]

    def npy(t):
        return t.detach().cpu().numpy().astype(np.float64)

    def base_conv(mod, inp):
        conv, bn = mod.conv, mod.bn
        if bn is None:                                  # a plain Conv2d (the head's predictions)
            w = npy(conv.weight).astype(np.float32)
            b = (npy(conv.bias) if conv.bias is not None else np.zeros(w.shape[0])).astype(np.float32)
            act = "none"
        else:                                           # BatchNorm folded into the weights, bias
            s = npy(bn.weight) / np.sqrt(npy(bn.running_var) + bn.eps)
            w = (npy(conv.weight) * s[:, None, None, None]).astype(np.float32)
            b = npy(bn.bias) - npy(bn.running_mean) * s
            if conv.bias is not None:
                b = b + npy(conv.bias) * s
            b = b.astype(np.float32)
            act = "silu" if isinstance(mod.act, nn.SiLU) else None
            if act is None:
                raise NotImplementedError(f"activation {type(mod.act).__name__}")
        xi = inp[0].permute(1, 2, 0).contiguous().numpy()
        H, W, Cin = xi.shape
        K, st, pad, g = conv.kernel_size[0], conv.stride[0], conv.padding[0], conv.groups
        Cout = w.shape[0]
        if g == 1:
            OH, OW, *_ = C.geometry(H, W, Cin, K, K, st, pad, cfg.D)
            stats[f"device conv{K}x{K}"] += 1
            stats["device MACs"] += OH * OW * Cout * Cin * K * K
            if a.backend == "ref":
                y = C.reference(xi, w, b, st, pad, act)
            else:
                y = launch(C.conv2d, x=Input(C.pad_input(xi, K, K, st, pad, cfg.D)),
                           w=Weight(C.pack_conv_weight(w, cfg.D), fmt="int8"), b=Input(b),
                           y=Output((OH * OW, Cout)), H=H, W=W, Cin=Cin, Cout=Cout, KH=K, KW=K,
                           stride=st, pad=pad, act=act)
        elif g == Cin == Cout:
            OH, OW, *_ = C.dw_geometry(H, W, Cin, K, st, pad)
            stats["device depthwise"] += 1
            stats["device MACs"] += OH * OW * Cin * K * K
            if a.backend == "ref":
                y = C.reference_dw(xi, w, b, st, pad, act)
            else:
                y = launch(C.depthwise2d, x=Input(C.pad_input_dw(xi, pad)),
                           w=Input(C.pack_dw_weight(w)), b=Input(b), y=Output((OH * OW, Cin)),
                           H=H, W=W, C=Cin, K=K, stride=st, pad=pad, act=act)
        else:
            raise NotImplementedError(f"groups {g}")
        return torch.from_numpy(np.asarray(y, np.float32).reshape(OH, OW, Cout)).permute(2, 0, 1)[None]

    def maxpool(mod, inp):
        xi = inp[0].permute(1, 2, 0).contiguous().numpy()
        H, W, Cc = xi.shape
        K, st, pad = mod.kernel_size, mod.stride, mod.padding
        stats["device maxpool"] += 1
        if a.backend == "ref":
            xp = np.full((H + 2 * pad, W + 2 * pad, Cc), -np.inf, np.float32)
            xp[pad:pad + H, pad:pad + W] = xi
            y = np.max([xp[i:i + H, j:j + W] for i in range(K) for j in range(K)], axis=0)
        else:
            y = launch(C.maxpool2d, x=Input(C.pad_input_pool(xi, pad)), y=Output((H * W, Cc)),
                       H=H, W=W, C=Cc, K=K, stride=st, pad=pad)
        return torch.from_numpy(np.asarray(y, np.float32).reshape(H, W, Cc)).permute(2, 0, 1)[None]

    def upsample(mod, inp):
        if mod.scale_factor != 2 or mod.mode != "nearest":
            raise NotImplementedError(f"upsample {mod.scale_factor} {mod.mode}")
        xi = inp[0].permute(1, 2, 0).contiguous().numpy()
        H, W, Cc = xi.shape
        stats["device upsample"] += 1
        if a.backend == "ref":
            y = xi.repeat(2, 0).repeat(2, 1)
        else:
            y = launch(C.upsample2x, x=Input(xi.reshape(H, W * Cc)), y=Output((4 * H * W, Cc)),
                       H=H, W=W, C=Cc)
        return torch.from_numpy(np.asarray(y, np.float32).reshape(2 * H, 2 * W, Cc)).permute(2, 0, 1)[None]

    dev = copy.deepcopy(ref)                    # get_model() returns the exp's cached model
    inner = {id(m.conv) for m in dev.modules() if isinstance(m, BaseConv)}
    for n, mod in dev.named_modules():
        if isinstance(mod, BaseConv):
            mod.forward = types.MethodType(lambda self, t: base_conv(self, t), mod)
        elif isinstance(mod, nn.Conv2d) and id(mod) not in inner:
            mod.forward = types.MethodType(
                lambda self, t: base_conv(types.SimpleNamespace(conv=self, bn=None), t), mod)
        elif isinstance(mod, nn.MaxPool2d):
            mod.forward = types.MethodType(lambda self, t: maxpool(self, t), mod)
        elif isinstance(mod, nn.Upsample):
            mod.forward = types.MethodType(lambda self, t: upsample(self, t), mod)

    outs = {"ref": {}, "dev": {}}
    for tag, model in (("ref", ref), ("dev", dev)):
        for n, mod in model.named_modules():
            if isinstance(mod, BaseConv):
                mod.register_forward_hook(
                    lambda m, i, o, n=n, tag=tag: outs[tag].__setitem__(n, o.detach().numpy().copy()))

    class HostOps(torch.overrides.TorchFunctionMode):
        """Counts the torch functions the device run executes on the host (outside the kernels)."""
        def __torch_function__(self, func, types_, args=(), kwargs=None):
            name = getattr(func, "__name__", str(func))
            if name not in ("__get__", "size", "dim", "contiguous", "permute", "numpy", "detach",
                            "__getitem__", "__setitem__", "copy_", "shape", "is_contiguous",
                            "cpu", "astype", "from_numpy", "__len__"):
                stats["host " + name] += 1
            return func(*args, **(kwargs or {}))

    with torch.no_grad():
        t0 = time.time()
        yr = ref(x).numpy()[0]
        t_ref = time.time() - t0
        t0 = time.time()
        with HostOps():
            yd = dev(x).numpy()[0]
        t_run = time.time() - t0

    print(f"\nhost {t_ref:.1f}s (torch), device run {t_run:.1f}s ({t_dev[0]:.1f}s in the simulator)")
    print("operators:")
    for k in sorted(stats):
        print(f"  {k:28s} {stats[k]}")
    errs = sorted(((rel(outs['dev'][n], outs['ref'][n]), n) for n in outs["ref"]), reverse=True)
    print(f"\nBaseConv outputs vs float: median rel L2 {np.median([e for e, _ in errs]):.4f}, worst:")
    for e, n in errs[:5]:
        print(f"  {e:.4f}  {n}")
    print(f"head output [anchors {yr.shape[0]}, 85]: box rel L2 {rel(yd[:, :4], yr[:, :4]):.4f}, "
          f"objectness max|d| {np.abs(yd[:, 4] - yr[:, 4]).max():.4f}, "
          f"class-score max|d| {np.abs(yd[:, 5:] - yr[:, 5:]).max():.4f}")
    dr, dd = detect(yr), detect(yd)
    print(f"\ndetections (score > 0.3): float {len(dr)}, device {len(dd)}")
    for tag, det in (("float", dr), ("device", dd)):
        for x0, y0, x1, y1, sc, c in det[:10]:
            print(f"  {tag:6s} class {int(c):2d} score {sc:.3f} box [{x0:.1f} {y0:.1f} {x1:.1f} {y1:.1f}]")
    match = [max((iou(p[:4], q[:4]) for q in dd if q[5] == p[5]), default=0.0) for p in dr]
    print(f"float detections matched by a device detection of the same class: IoU min "
          f"{min(match, default=0):.3f}, mean {np.mean(match) if match else 0:.3f}")
    return 0


PROG_GAP = 1 << 20


def run_on_board(board, comp, img):
    """Run one compiled kernel on the card: its DRAM image (up to its last placed byte) at 0,
    the program above it, then the output read back. Returns (y, run stats)."""
    from opentpu import quant as Q
    from opentpu.runtime import Input, Output, Weight, collect
    D = comp.cfg.D
    top = 0                     # the end of every placed argument: all of it is written, so the
    for a, addr in comp.layout.params.values():     # card never reads a beat it has not written
        if isinstance(a, Output):
            top = max(top, addr + 4 * int(np.prod(a.shape)))
        elif isinstance(a, Weight):
            n, k = a.array.shape
            top = max(top, addr[0] + n * Q.row_bytes(k, a.fmt, D), addr[1] + n * Q.scale_bytes(k, a.fmt, D))
        elif isinstance(a, Input):
            top = max(top, addr + a.array.nbytes)
        else:
            raise NotImplementedError(f"placing {type(a).__name__} on the card")
    top = -(-top // 4096) * 4096
    used = img[:top].copy()
    board.write(0, used)
    at = top + PROG_GAP
    board.load(at, comp.programs[0])
    st = board.run(timeout=60.0)
    dram = img.copy()
    for a, addr in comp.layout.params.values():
        if isinstance(a, Output):
            n = 4 * int(np.prod(a.shape))
            dram[addr:addr + n] = board.read(addr, n)
    return collect(comp, [dram], [None], st).outputs["y"], st


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def load_image(path, size):
    """YOLOX's ValTransform (legacy=False): letterbox resize to size x size, pad value 114, BGR
    channel order, values 0..255, CHW float32. Without a path: a fixed random image."""
    if path is None:
        return np.random.default_rng(0).uniform(0, 255, (3, size, size)).astype(np.float32)
    from PIL import Image
    im = Image.open(path).convert("RGB")
    r = min(size / im.height, size / im.width)
    im = im.resize((int(im.width * r), int(im.height * r)), Image.BILINEAR)
    out = np.full((size, size, 3), 114, np.float32)
    a = np.asarray(im, np.float32)[:, :, ::-1]                    # RGB -> BGR (cv2's order)
    out[:a.shape[0], :a.shape[1]] = a
    return np.ascontiguousarray(out.transpose(2, 0, 1))


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def detect(y, conf=0.3, nms=0.45):
    """YOLOX postprocess (class-aware NMS) in numpy: rows [x0, y0, x1, y1, score, class]."""
    cls = y[:, 5:].argmax(1)
    score = y[:, 4] * y[np.arange(len(y)), 5 + cls]
    keep = score > conf
    b = y[keep, :4]
    boxes = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                      b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
    cand = sorted(zip(boxes.tolist(), score[keep].tolist(), cls[keep].tolist()), key=lambda t: -t[1])
    out = []
    for bx, sc, c in cand:
        if all(o[5] != c or iou(bx, o[:4]) <= nms for o in out):
            out.append([*bx, sc, c])
    return out


if __name__ == "__main__":
    sys.exit(main())
