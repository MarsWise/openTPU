#!/usr/bin/env python3
"""Cycles of one TMM GRQ QUAD (Bonsai 2's decode matmul, M 1) on the RTL (Verilator, the board's
memory path), against its ideal: one 128-byte chunk a cycle (N/4 chunks a K block, KB blocks).

    python tools/gq4_bench.py [--n 1024] [--kb 20] [--mcols 8] [--lat 60] [--ldc]

Prints the command's cycles (the program with it minus the program without it), the chunks, the
cycles a K block over its chunks, and checks the result against the ISA simulator bit for bit.
--ldc runs the card's controller model (LiteDRAM, DDR3-1066) instead of the native memory model.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opentpu import isa as I, rtlsim                                    # noqa: E402
from opentpu.isasim import Config, Machine                              # noqa: E402
from opentpu.quant import grq_scales, pack_ternary_grq                  # noqa: E402


def build(n, kb, mcols, seed=0):
    cfg = Config(S=1, D=128, MCOLS=mcols, ACT_ROWS=2 * mcols, ACT_BLOCKS=max(16, kb), TGSC=True,
                 TGRQ=True, TQUAD=True, TPAIR=True, PAIR=True, TGQ4=True, DRAM_BYTES=1 << 24)
    D = cfg.D
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(1, kb * D)).astype(np.float32)
    w = rng.integers(-1, 2, (n, kb * D), dtype=np.int8)
    scales = rng.uniform(.001, .2, (n, kb)).astype(np.float16).astype(np.float32)
    planes = pack_ternary_grq(w, D)
    splanes, srs = grq_scales(scales)
    rs = planes.shape[1]                              # planes back to back, as the image packs them
    W, S = 0x100000, 0x100000 + kb * rs + 4096
    image = np.zeros(cfg.DRAM_BYTES, np.uint8)
    image[:x.nbytes] = x.view(np.uint8).reshape(-1)
    for k in range(kb):
        image[W + k * rs:W + (k + 1) * rs] = planes[k]
        image[S + k * srs:S + (k + 1) * srs] = splanes[k].view(np.uint8)
    out = 8192
    head = [I.ld(0, 0, x.size), I.qact(0, 1, 0, kb, kb * D, dup=True, dup4=True)]
    quad = I.tmm(W, S, out, n, kb, rs, n, 1, 0, srs, gsc=True, grq=True, quad=True)
    return cfg, image, head, quad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--kb", type=int, default=20)
    ap.add_argument("--mcols", type=int, default=8)
    ap.add_argument("--lat", type=int, default=60, help="native model's read latency (cycles)")
    ap.add_argument("--ldc", action="store_true", help="the card's LiteDRAM controller model")
    ap.add_argument("--pipe", type=int, default=1, choices=[0, 1],
                    help="otpu_tmm_gsc PIPE (0: a K block's subcommand after the one before completed)")
    a = ap.parse_args()
    cfg, image, head, quad = build(a.n, a.kb, a.mcols)
    kw = dict(axi=True, stall=0, bw=100, lat=a.lat) if not a.ldc else dict(axi=True, ldc=1066)
    kw["uarch"] = {"GSC_PIPE": a.pipe}
    progs = {"base": head + [I.halt()], "quad": head + [quad, I.halt()]}
    cyc = {}
    for name, p in progs.items():
        drams, tmems, st = rtlsim.run(cfg, [p], [image.copy()], **kw)
        cyc[name] = st["cycles"][0] if isinstance(st["cycles"], list) else st["cycles"]
        if name == "quad":
            ref = Machine(cfg, [p], [image.copy()]).run().slices[0]
            np.testing.assert_array_equal(tmems[0], ref.tmem)
    c = cyc["quad"] - cyc["base"]
    chunks = a.n // 4 * a.kb
    print(f"GRQ QUAD N {a.n} KB {a.kb} MCOLS {a.mcols} PIPE {a.pipe} "
          f"({'LiteDRAM' if a.ldc else f'lat {a.lat}'}): "
          f"{c} cycles, {chunks} chunks ({c / chunks:.2f} cycles a chunk), "
          f"{(c - chunks) / a.kb:.0f} extra cycles a K block; RTL == ISA")


if __name__ == "__main__":
    main()
