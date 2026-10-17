"""A real BitNet b1.58 2B4T projection on the board, against the HF AutoBitLinear reference.

Usage: python tools/bitnet_layer_check.py PACKED.npy SCALE [--m 8] [--outliers]
PACKED.npy is the checkpoint's uint8 [N/4, K] weight, SCALE its weight_scale. The board result
must equal the ISA simulator bit for bit; its distance to the float HF reference is reported
(QACT quantizes per D-block with a power-of-two scale, HF per token with absmax/127).
"""
import argparse
import sys

import numpy as np

from opentpu import isa as I
from opentpu.bitnet import hf_to_tmm, reference_linear, unpack_hf_bitnet
from opentpu.host.board import Board, XdmaTransport, device_config
from opentpu.host.checks import PROG_AT, run_demo
from opentpu.host.runstate import busy_exits


@busy_exits
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("packed")
    ap.add_argument("scale", type=float)
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--outliers", action="store_true", help="a few x20 channels in the input")
    a = ap.parse_args()
    hf = np.load(a.packed)
    board = Board(XdmaTransport(a.dev), check=False)
    cfg = device_config(board.info(), DRAM_BYTES=1 << 23)
    D, M = cfg.D, a.m
    N, K = hf.shape[0] * 4, hf.shape[1]
    tmm_w = hf_to_tmm(hf, D)
    KB = K // D
    rng = np.random.default_rng(7)
    x = rng.standard_normal((M, K), dtype=np.float32)
    if a.outliers:
        x[:, rng.choice(K, 8, replace=False)] *= 20
    weight_at, scale_at, out_at, store_at = 0x100000, 0x300000, M * K, 0x500000
    assert M * (K + N) <= cfg.TMEM_WORDS and weight_at + tmm_w.size <= scale_at
    img = np.zeros(PROG_AT, np.uint8)
    img[:x.nbytes] = x.view(np.uint8).reshape(-1)
    img[weight_at:weight_at + tmm_w.size] = tmm_w.reshape(-1)
    sv = np.full(N, a.scale, np.float32)
    img[scale_at:scale_at + sv.nbytes] = sv.view(np.uint8)
    prog = [I.ld(0, 0, M * K), I.qact(0, M, 0, KB, K),
            I.tmm(weight_at, scale_at, out_at, N, KB, tmm_w.shape[1], N, M, 0, 4, rsc=True),
            I.st(store_at, out_at, M * N), I.halt()]
    ok, msg, st = run_demo(board, cfg, prog, img)
    if not ok:
        sys.exit(f"MISMATCH vs ISA simulator: {msg}")
    got = board.read(store_at, 4 * M * N).view("<f4").reshape(M, N)
    want = reference_linear(x, hf, a.scale)
    ideal = x @ unpack_hf_bitnet(hf).T.astype(np.float32) * np.float32(a.scale)
    rel = np.linalg.norm(got - want, axis=1) / np.linalg.norm(want, axis=1)
    rel_hw = np.linalg.norm(got - ideal, axis=1) / np.linalg.norm(ideal, axis=1)
    rel_hf = np.linalg.norm(want - ideal, axis=1) / np.linalg.norm(ideal, axis=1)
    print(f"BitNet projection N={N} K={K} M={M} outliers={a.outliers}: bit-exact with ISA "
          f"simulator, {st['cycles']} cycles")
    print(f"vs HF reference: rel err per token max {rel.max():.4f} mean {rel.mean():.4f}; "
          f"max abs err {np.abs(got - want).max():.4f} (output absmax {np.abs(want).max():.2f})")
    print(f"vs unquantized float: board max {rel_hw.max():.4f}, HF reference {rel_hf.max():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
