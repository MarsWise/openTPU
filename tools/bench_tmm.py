"""Ternary TMM throughput on the board: real-size layers, checked against the ISA simulator.

Usage: python tools/bench_tmm.py [--dev /dev/xdma0] [--k 2048] [--n 1024] [--m 16]
Reports measured cycles for the TMM program (after subtracting an identical program without the
TMM) and MAC/cycle, so loads and QACT are not counted as TMM time.
"""
import argparse
import sys
import time

import numpy as np

from opentpu import isa as I
from opentpu.host.board import Board, XdmaTransport, device_config
from opentpu.host.runstate import busy_exits
from opentpu.host.checks import PROG_AT, run_demo
from opentpu.quant import pack_ternary_mxu


def build(cfg, M, K, N, tmm=True):
    D = cfg.D
    KB = K // D
    rng = np.random.default_rng(K + N + M)
    values = rng.standard_normal((M, K), dtype=np.float32)
    weights = rng.choice(np.array([-1, 0, 1], np.int8), (N, K))
    scales = rng.uniform(0.05, 0.2, N).astype(np.float32)
    packed = pack_ternary_mxu(weights, D)
    weight_at, scale_at, out_at = 0x100000, 0x300000, 0
    stride, ostride = packed.shape[1], N
    top = max(weight_at + packed.size, scale_at + scales.nbytes)
    assert top <= PROG_AT, "layer does not fit below the program area"
    img = np.zeros(PROG_AT, np.uint8)
    img[:values.nbytes] = values.view(np.uint8).reshape(-1)
    img[weight_at:weight_at + packed.size] = packed.reshape(-1)
    img[scale_at:scale_at + scales.nbytes] = scales.view(np.uint8)
    act_at = 0
    out_at = M * K
    assert out_at + M * ostride <= cfg.TMEM_WORDS, "activations plus outputs exceed TMEM"
    prog = [I.ld(act_at, 0, M * K), I.qact(0, M, 0, KB, K)]
    if tmm:
        prog.append(I.tmm(weight_at, scale_at, out_at, N, KB, stride, ostride, M, 0, 4,
                          rsc=True))
    prog.append(I.halt())
    return prog, img, (values, weights, scales, out_at, ostride)


@busy_exits
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--n", type=int, default=1024)
    a = ap.parse_args()
    board = Board(XdmaTransport(a.dev), check=False)
    info = board.info()
    cfg = device_config(info, DRAM_BYTES=1 << 23)
    M, K, N = a.m, a.k, a.n
    if K % cfg.D or K // cfg.D > cfg.ACT_BLOCKS:
        sys.exit(f"K must be a multiple of {cfg.D} and at most {cfg.ACT_BLOCKS * cfg.D}")
    if M > cfg.ACT_ROWS:
        sys.exit(f"M must be at most {cfg.ACT_ROWS}")
    prog, img, (values, weights, scales, out_at, ostride) = build(cfg, M, K, N)
    ok, msg, st = run_demo(board, cfg, prog, img)
    if not ok:
        sys.exit(f"MISMATCH vs ISA simulator: {msg}")
    base_prog, base_img, _ = build(cfg, M, K, N, tmm=False)
    ok0, msg0, st0 = run_demo(board, cfg, base_prog, base_img)
    # read the result: store TMEM to DRAM and compare with the float reference
    out_words = M * ostride
    rprog = prog[:-1] + [I.st(0x500000, out_at, out_words), I.halt()]
    ok2, msg2, _ = run_demo(board, cfg, rprog, img)
    if not ok2:
        sys.exit(f"MISMATCH (with store): {msg2}")
    res = board.read(0x500000, 4 * out_words).view("<f4").reshape(M, ostride)
    want = values @ weights.T.astype(np.float32) * scales
    err = float(np.linalg.norm(res - want) / np.linalg.norm(want))
    tmm_cycles = st["cycles"] - st0["cycles"]
    macs = M * N * K
    khz = info.get("core_khz", 133333)
    print(f"TMM M={M} K={K} N={N}: bit-exact with ISA simulator, rel err vs float {err:.4f}")
    print(f"total cycles {st['cycles']}, without TMM {st0['cycles']}, TMM {tmm_cycles}")
    print(f"{macs / tmm_cycles:.1f} MAC/cycle, {2 * macs / tmm_cycles * khz * 1e3 / 1e9:.2f} "
          f"GOPS at {khz / 1000:.2f} MHz")
    wbytes = N * ((K // 4 + cfg.D - 1) // cfg.D * cfg.D)
    print(f"weight bytes {wbytes}, {wbytes / tmm_cycles:.1f} B/cycle "
          f"({wbytes / tmm_cycles * khz * 1e3 / 1e9:.2f} GB/s)")
    return 0 if err < 0.03 else 1


if __name__ == "__main__":
    sys.exit(main())
