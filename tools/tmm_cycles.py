"""TMM cycles in the RTL (Verilator) at M = 1, 2, 4 rows: the measurement behind
docs/nextgen/HETEROGENEOUS_CORE_DSE.md C1 (TMM K-split). Usage: python tools/tmm_cycles.py"""
import numpy as np
from opentpu import Config, rtlsim
from opentpu import isa as I
from opentpu.quant import pack_ternary_mxu
DATA = 0
cfg = Config(S=1, D=128, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=32, DRAM_BYTES=1 << 24)
D = cfg.D
rng = np.random.default_rng(0)
def cyc(M, N, KB, tmm=True):
    values = rng.standard_normal((M, KB * D), dtype=np.float32)
    w = rng.choice(np.array([-1, 0, 1], np.int8), (N, KB * D)); packed = pack_ternary_mxu(w, D)
    wa, sa, oa = 0x100000, 0x300000, 8192
    img = np.zeros(cfg.DRAM_BYTES, np.uint8)
    img[DATA:DATA + values.nbytes] = values.view(np.uint8).reshape(-1)
    img[wa:wa + packed.size] = packed.reshape(-1)
    img[sa:sa + 4 * N] = np.full(N, 0.1, np.float32).view(np.uint8)
    prog = [I.ld(DATA, 0, M * KB * D), I.qact(0, M, 0, KB, KB * D)]
    if tmm:
        prog.append(I.tmm(wa, sa, oa, N, KB, packed.shape[1], N, M, 0, 4, unit=False, acc=False, rsc=True))
    prog.append(I.halt())
    _, _, st = rtlsim.run(cfg, [prog], [img.copy()])
    return st["cycles"]
for M in (1, 2, 4):
    for N, KB in ((64, 20), (128, 20)):
        c = cyc(M, N, KB) - cyc(M, N, KB, tmm=False)
        print(f"M={M} N={N} K={KB*D}: TMM {c} cycles, {M*N*KB*D/c:.1f} MAC/cycle")
