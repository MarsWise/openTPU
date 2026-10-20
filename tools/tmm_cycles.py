"""TMM cycles in the RTL (Verilator) at M = 1, 2, 4 ACT rows, plain and TMM PAIR (the ternary
K-split by two): the measurement behind docs/nextgen/HETEROGENEOUS_CORE_DSE.md C1.
Usage: python tools/tmm_cycles.py [--n 128] [--kb 20]"""
import argparse

import numpy as np

from opentpu import Config, rtlsim
from opentpu import isa as I
from opentpu.quant import pack_ternary_mxu


def cycles(cfg, M, N, KB, pair, tmm=True, seed=0):
    D = cfg.D
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((M, KB * D), dtype=np.float32)
    packed = pack_ternary_mxu(rng.choice(np.array([-1, 0, 1], np.int8), (N, KB * D)), D)
    wa, sa, oa = 0x100000, 0x300000, 8192
    img = np.zeros(cfg.DRAM_BYTES, np.uint8)
    img[:x.nbytes] = x.view(np.uint8).reshape(-1)
    img[wa:wa + packed.size] = packed.reshape(-1)
    img[sa:sa + 4 * N] = np.full(N, 0.1, np.float32).view(np.uint8)
    prog = [I.ld(0, 0, M * KB * D), I.qact(0, M, 0, KB, KB * D, dup=pair)]
    if tmm:
        prog.append(I.tmm(wa, sa, oa, N, KB, packed.shape[1], N, M, 0, 4, rsc=True, pair=pair))
    prog.append(I.halt())
    return rtlsim.run(cfg, [prog], [img])[2]["cycles"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--kb", type=int, default=20)
    a = ap.parse_args()
    cfg = Config(S=1, D=128, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=32, DRAM_BYTES=1 << 24)
    K = a.kb * cfg.D
    for M in (1, 2, 4):
        for pair in ((False, True) if 2 * M <= cfg.MCOLS else (False,)):
            c = cycles(cfg, M, a.n, a.kb, pair) - cycles(cfg, M, a.n, a.kb, pair, tmm=False)
            print(f"M={M} N={a.n} K={K}{' PAIR' if pair else '     '}: TMM {c} cycles, "
                  f"{M * a.n * K / c:.1f} MAC/cycle")


if __name__ == "__main__":
    main()
