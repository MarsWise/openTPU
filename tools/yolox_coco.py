"""COCO val2017 mAP of YOLOX-Nano: the float model, an int8 emulation of the device's numerics,
and the device itself (the whole graph on the card, tools/yolox_graph.py's plan).

Usage: python tools/yolox_coco.py YOLOX_REPO yolox_nano.pth COCO_DIR [--backend float|emul|board]
                                  [--limit N] [--size 416] [--dev /dev/xdma0] [--stubs DIR]
COCO_DIR holds val2017/ and annotations/instances_val2017.json. Preprocessing and evaluation as
YOLOX's: letterbox to size x size (pad 114, BGR), test_conf 0.01, class-aware NMS 0.65, boxes
scaled back, pycocotools COCOeval (bbox). --limit N evaluates the first N image ids (sorted).
`emul`: the PyTorch model with every groups=1 convolution's weights replaced by the int8 MXU
rows' dequantized values (quant.quantize_mxu / dequantize_mxu, as the card stores them) and its
input rows quantized per pixel as QACT does (int8, power-of-two offsets per D/4 group): fast,
close to the card's numerics, not bit-exact (the card quantizes each Kp-wide patch).
"""
import argparse
import copy
import json
import sys
import time
import types

import numpy as np


def nms(boxes, scores, thr):
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        r = order[1:]
        xx0 = np.maximum(boxes[i, 0], boxes[r, 0]); yy0 = np.maximum(boxes[i, 1], boxes[r, 1])
        xx1 = np.minimum(boxes[i, 2], boxes[r, 2]); yy1 = np.minimum(boxes[i, 3], boxes[r, 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        b = (boxes[r, 2] - boxes[r, 0]) * (boxes[r, 3] - boxes[r, 1])
        order = r[inter / (a + b - inter + 1e-12) <= thr]
    return keep


def postprocess(y, conf=0.01, thr=0.65):
    """YOLOX postprocess on decoded rows [anchors, 85]: boxes xyxy, score, class."""
    cls = y[:, 5:].argmax(1)
    score = y[:, 4] * y[np.arange(len(y)), 5 + cls]
    k = score >= conf
    y, cls, score = y[k], cls[k], score[k]
    boxes = np.stack([y[:, 0] - y[:, 2] / 2, y[:, 1] - y[:, 3] / 2,
                      y[:, 0] + y[:, 2] / 2, y[:, 1] + y[:, 3] / 2], 1)
    out = []
    for c in np.unique(cls):
        idx = np.nonzero(cls == c)[0]
        for j in nms(boxes[idx], score[idx], thr):
            out.append((*boxes[idx[j]], score[idx[j]], c))
    return out


def letterbox(path, size):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    r = min(size / im.height, size / im.width)
    im2 = im.resize((int(im.width * r), int(im.height * r)), Image.BILINEAR)
    out = np.full((size, size, 3), 114, np.float32)
    a = np.asarray(im2, np.float32)[:, :, ::-1]
    out[:a.shape[0], :a.shape[1]] = a
    return np.ascontiguousarray(out.transpose(2, 0, 1)), r


def emulate(model, D=128):
    """Patch the model's groups=1 convolutions with the card's int8 numerics (see above)."""
    import torch
    from opentpu import fp32 as F, quant as Q
    from opentpu.kernels import conv as C
    m = copy.deepcopy(model)
    for mod in m.modules():
        if isinstance(mod, torch.nn.Conv2d) and mod.groups == 1:
            w = mod.weight.detach().numpy()
            Co, Ci, K, _ = w.shape
            p = C.pack_conv_weight(w, D)
            q, sc = Q.quantize_mxu(p, "int8", D)
            deq = Q.dequantize_mxu(q, sc, "int8", p.shape[1], D)
            Kp = p.shape[1] // K
            wd = deq.reshape(Co, K, Kp)[:, :, :K * Ci].reshape(Co, K, K, Ci).transpose(0, 3, 1, 2)
            mod.weight.data = torch.from_numpy(np.ascontiguousarray(wd, np.float32))
            orig = mod.forward

            def fwd(self, t, orig=orig):
                xn = t[0].permute(1, 2, 0).numpy()
                H, W, Cc = xn.shape
                v = np.pad(xn.reshape(-1, Cc), ((0, 0), (0, (-Cc) % D))).astype(np.float32)
                q8, s, d = F.quantize_mx(v, D // 4, False)
                rec = (q8.astype(np.float64).reshape(len(v), -1, D // 4)
                       * (s[:, None, None] / np.ldexp(1.0, d.astype(int))[:, :, None]))
                rec = rec.reshape(len(v), -1)[:, :Cc].reshape(H, W, Cc).astype(np.float32)
                return orig(torch.from_numpy(rec).permute(2, 0, 1)[None])
            mod.forward = types.MethodType(fwd, mod)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("weights")
    ap.add_argument("coco")
    ap.add_argument("--backend", default="float", choices=("float", "emul", "board"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--size", type=int, default=416)
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--stubs", default=None)
    ap.add_argument("--out", default=None, help="write the detections JSON here")
    a = ap.parse_args()
    sys.path.insert(0, a.repo)
    if a.stubs:
        sys.path.insert(0, a.stubs)
    import torch
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from yolox.exp import get_exp
    exp = get_exp(f"{a.repo}/exps/default/yolox_nano.py", None)
    model = exp.get_model().eval()
    model.load_state_dict(torch.load(a.weights, map_location="cpu", weights_only=False)["model"])
    gt = COCO(f"{a.coco}/annotations/instances_val2017.json")
    cats = sorted(gt.getCatIds())
    ids = sorted(gt.getImgIds())
    if a.limit:
        ids = ids[:a.limit]
    run = make_runner(a, model, exp)
    dets, t0 = [], time.time()
    for i, iid in enumerate(ids):
        info = gt.loadImgs(iid)[0]
        x, r = letterbox(f"{a.coco}/val2017/{info['file_name']}", a.size)
        for x0, y0, x1, y1, s, c in postprocess(run(x)):
            dets.append({"image_id": iid, "category_id": cats[int(c)], "score": float(s),
                         "bbox": [float(x0 / r), float(y0 / r), float((x1 - x0) / r),
                                  float((y1 - y0) / r)]})
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(ids)} images, {time.time() - t0:.0f}s", flush=True)
    if a.out:
        json.dump(dets, open(a.out, "w"))
    print(f"backend {a.backend}: {len(ids)} images, {len(dets)} detections, "
          f"{(time.time() - t0) / max(len(ids), 1):.2f} s/image")
    if not dets:
        print("no detections")
        return 1
    ev = COCOeval(gt, gt.loadRes(dets), "bbox")
    ev.params.imgIds = ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    print(f"RESULT backend={a.backend} images={len(ids)} mAP50-95={ev.stats[0]:.4f} "
          f"mAP50={ev.stats[1]:.4f}")
    return 0


def make_runner(a, model, exp):
    import torch
    if a.backend in ("float", "emul"):
        m = model if a.backend == "float" else emulate(model)
        m.head.decode_in_inference = True

        def run(x):
            with torch.no_grad():
                return m(torch.from_numpy(x)[None])[0].numpy()
        return run
    # the whole graph on the card, decode included (yolox_graph --strict-fpga's plan)
    import dataclasses
    import torch.fx as fx
    import torch.nn as nn
    from yolox.models.network_blocks import BaseConv, Focus
    from opentpu import cnngraph as G
    from opentpu.host.board import Board, XdmaTransport, device_config
    from opentpu.compiler import Affine, Tensor
    from opentpu.kernels import detect as K

    class Leaf(fx.Tracer):
        def is_leaf_module(self, mod, qn):
            return isinstance(mod, (BaseConv, Focus, nn.MaxPool2d, nn.Upsample, nn.Conv2d)) or \
                super().is_leaf_module(mod, qn)
    model.head.decode_in_inference = False
    gm = fx.GraphModule(model, Leaf().trace(model))
    board = Board(XdmaTransport(a.dev), check=False)
    cfg = dataclasses.replace(device_config(board.info(), DRAM_BYTES=1 << 28), S=1)
    dec = {}

    def extend(pl):
        lv = [n for n in gm.graph.nodes if isinstance(pl.env.get(n), G.Val) and
              any(pl.env.get(u) is None for u in n.users)]
        sizes = [(pl.env[n].buf.H, pl.env[n].buf.W, pl.env[n].C) for n in lv]
        cols = sizes[0][2]
        out = pl.buf("decoded", sum(h * w for h, w, _ in sizes), 1, cols)
        r0 = 0
        for n, (h, w, c) in zip(lv, sizes):
            g, v = pl.const(f"grid{h}x{w}", K.grid(h, w)), pl.env[n]

            def bound(p, v=v, g=g, h=h, w=w, r0=r0):
                return dict(x=Tensor(Affine(v.buf.addr), (h * w, cols), (cols, 1)),
                            g=Tensor(Affine(g[2]), (h * w, 2), (2, 1)),
                            y=Tensor(Affine(out.addr + 4 * r0 * cols), (h * w, cols), (cols, 1)),
                            n=h * w, cols=cols, stride=a.size // h)
            pl.kernel_op(f"decode{h}x{w}", K.yolox_decode, bound, G.Val(out))
            r0 += h * w
        dec["val"] = G.Val(out)

    plan = G.build(gm, (3, a.size, a.size), cfg, extend)
    img = plan.image(plan.size, cfg.D)
    board.write(0, img)
    at = -(-(len(img) + (1 << 20)) // 4096) * 4096
    xb = plan.inputs["x"]
    ob = dec["val"].buf

    def run(x):
        plan.put_input(img, "x", x.transpose(1, 2, 0))
        board.write(xb.addr, img[xb.addr:xb.addr + xb.nbytes])
        for op in plan.ops:
            board.load(at, op.prog)
            board.run(timeout=60.0)
        raw = board.read(ob.addr, ob.nbytes).view(np.float32)
        return raw.reshape(ob.H, ob.C)
    return run


if __name__ == "__main__":
    sys.exit(main())
