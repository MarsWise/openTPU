"""Fixed-count, class-aware NMS as an FPGA program.

Input consists of six fp32 vectors in DRAM: x1, y1, x2, y2, non-negative
scores, and integral class identifiers. Output is a fixed number of rows
[x1,y1,x2,y2,score,class,valid]; valid is 1 iff score > score_threshold.
Coordinates must be finite and ordered; IoU suppression is strict and uses
intersection > iou_threshold * union in the device's fp32 arithmetic.

This is a standalone primitive, not yet integrated into the YOLO graph's
class-score reduction and box-format conversion.
"""
from __future__ import annotations

import math
from .. import isa as I


def yolox_preprocess(n: int, cols: int, source: int, destination: int, *, tmem_words: int,
                     chunk_rows: int = 256):
    """Turn decoded YOLOX [n, 5+classes] into the six DRAM vectors for NMS.

    Converts center/size boxes to corners, selects the highest class per row,
    and multiplies its probability by objectness on the VPU. No host tensor
    computation is involved. The program uses bounded TMEM tiles.
    """
    if not 0 < n < 65536 or not 6 <= cols < 65536 or chunk_rows <= 0:
        raise ValueError("invalid YOLOX dimensions")
    if min(source, destination) < 0 or source % 4 or destination % 4:
        raise ValueError("YOLOX DRAM addresses must be non-negative and word aligned")
    if source + 4 * n * cols > 1 << 32 or destination + 24 * n > 1 << 32:
        raise ValueError("YOLOX DRAM region exceeds the 32-bit address space")
    if max(source, destination) < min(source + 4 * n * cols, destination + 24 * n):
        raise ValueError("YOLOX input and output must not overlap")
    batch = min(chunk_rows, tmem_words // (cols + 10))
    if not batch:
        raise ValueError("not enough TMEM for a YOLOX row")
    p = []
    for r0 in range(0, n, batch):
        m = min(batch, n - r0)
        half_x, half_y = batch * cols, batch * (cols + 1)
        vectors = [batch * (cols + 2 + j) for j in range(6)]
        pair = batch * (cols + 8)
        p.append(I.ld(source + 4 * r0 * cols, 0, m * cols))
        for half, coordinate in ((half_x, 2), (half_y, 3)):
            p.append(I.vop(I.V_MUL, half, coordinate, 0, m, 1, 1, cols, 0,
                           I.B_SCALAR, 0.5))
        for out, center, half, op in ((vectors[0], 0, half_x, I.V_SUB),
                                       (vectors[1], 1, half_y, I.V_SUB),
                                       (vectors[2], 0, half_x, I.V_ADD),
                                       (vectors[3], 1, half_y, I.V_ADD)):
            p.append(I.vop(op, out, center, half, m, 1, 1, cols, 1, I.B_ROW))
        p += [I.argmax(pair, 5, m, cols - 5, drs=2, ars=cols),
              I.vop(I.V_MUL, vectors[4], 4, pair, m, 1, 1, cols, 2, I.B_ROW),
              I.vop(I.V_COPY, vectors[5], pair + 1, 0, m, 1, 1, 2, 0)]
        p += [I.st(destination + 4 * (j * n + r0), vectors[j], m) for j in range(6)]
    return p


def program(n: int, count: int, source: int, destination: int, *, tmem_words: int,
            imem_words: int = 32768, score_threshold: float = 0.3,
            iou_threshold: float = 0.65):
    """Make a no-host-arithmetic NMS program for an r6 VCMP configuration.

    The caller must load six contiguous vectors of n floats at `source`. The
    source and destination DRAM regions must not overlap. Only the final fixed
    result buffer needs to be read by the host. The current implementation
    unrolls `count` iterations; `imem_words` is the target's instruction-memory size.
    """
    if not 0 < n < 65536 or not 0 < count <= n:
        raise ValueError("NMS requires 1 <= count <= n < 65536")
    if not all(math.isfinite(x) for x in (score_threshold, iou_threshold)) or \
            score_threshold < 0 or not 0 <= iou_threshold <= 1:
        raise ValueError("NMS thresholds must be finite; score >= 0 and IoU in [0,1]")
    if source < 0 or destination < 0 or source % 4 or destination % 4:
        raise ValueError("NMS DRAM addresses must be non-negative and word aligned")
    if source + 24 * n > 1 << 32 or destination + 28 * count > 1 << 32:
        raise ValueError("NMS DRAM region exceeds the 32-bit address space")
    if 8 * (8 + 27 * count) > imem_words:
        raise ValueError("NMS unrolled program does not fit instruction memory")
    if max(source, destination) < min(source + 24 * n, destination + 28 * count):
        raise ValueError("NMS source and result regions must not overlap")

    x1, y1, x2, y2, score, label, area, t0, t1, t2, t3 = (i * n for i in range(11))
    result = 11 * n
    peak = result + 7 * count
    if peak + 2 > tmem_words:
        raise ValueError(f"NMS needs {peak + 2} TMEM words, only {tmem_words} available")
    p = [I.ld(source, 0, 6 * n)]

    def unary_scalar(f, dst, a, v):
        return I.vop(f, dst, a, 0, 1, n, n, n, 0, I.B_SCALAR, v)

    def full(f, dst, a, b):
        return I.vop(f, dst, a, b, 1, n, n, n, n, I.B_FULL)

    def selected(f, dst, a, b):
        # R1 indexes the selected element in operand B.
        return I.vop(f, dst, a, b, 1, n, n, n, 0, I.B_ROW, rc=1)

    p += [full(I.V_SUB, t0, x2, x1), unary_scalar(I.V_MAX, t0, t0, 0.0),
          full(I.V_SUB, t1, y2, y1), unary_scalar(I.V_MAX, t1, t1, 0.0),
          full(I.V_MUL, area, t0, t1)]
    for k in range(count):
        p += [I.argmax(peak, score, 1, n, drs=2),
              I.rld(1, peak + 1, bound=(0, n - 1))]
        for j, arr in enumerate((x1, y1, x2, y2, score, label)):
            p.append(I.vop(I.V_COPY, result + 7 * k + j, arr, 0, 1, 1, 1, 1, 0, rb=1))
        p.append(I.vop(I.V_CMP_GT, result + 7 * k + 6, result + 7 * k + 4, 0,
                       1, 1, 1, 1, 0, I.B_SCALAR, score_threshold))
        p += [selected(I.V_MIN, t0, x2, x2), selected(I.V_MAX, t1, x1, x1),
              full(I.V_SUB, t0, t0, t1), unary_scalar(I.V_MAX, t0, t0, 0.0),
              selected(I.V_MIN, t2, y2, y2), selected(I.V_MAX, t3, y1, y1),
              full(I.V_SUB, t2, t2, t3), unary_scalar(I.V_MAX, t2, t2, 0.0),
              full(I.V_MUL, t0, t0, t2), selected(I.V_ADD, t1, area, area),
              full(I.V_SUB, t1, t1, t0), unary_scalar(I.V_MUL, t1, t1, iou_threshold),
              full(I.V_CMP_GT, t0, t0, t1), selected(I.V_CMP_EQ, t1, label, label),
              full(I.V_MUL, t0, t0, t1), unary_scalar(I.V_RSUB, t1, t0, 1.0),
              full(I.V_MUL, score, score, t1),
              # A degenerate selected box has IoU 0 with itself, so clear it explicitly.
              I.vop(I.V_FILL, score, score, 0, 1, 1, 1, 1, 0, I.B_SCALAR, 0.0, ra=1)]
    p += [I.st(destination, result, 7 * count), I.halt()]
    return p
