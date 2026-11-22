"""The r8 instructions on the card (or the board model: --sim), each against the ISA simulator
bit for bit: the bitstream's identity (CFG_HASH, the r8 CAPS2 bits), otpu-diag's one-program-
per-instruction checks (every VOP function the bitstream has, MM / QACT / QST, gen, WAITW), and
r8's multi-instruction cases taken from the RTL tests (each the same program and DRAM image the
Verilator test runs):

  TMM CQ1..CQ4 (LUTSET), TMM GSC / GRQ, TMM T5 (dense trits), QST Q4, QST Q6 (6-bit K: three
  planes) and Q6T (6-bit V^T, groups of four tokens), VOP CSWP (a bitonic sort), VOP RAND,
  LDX (row gather)

A case runs from a cleared TMEM (TMEM keeps the previous run's contents on the card), and its
whole TMEM is stored to DRAM at the end: the comparison is DRAM [0, 0x280000) -- the case's own
image and outputs and the TMEM dump -- card against ISA.

    python tools/r8_card_check.py [--dev /dev/xdma0 | --sim] [--skip-ops]

Prints one CHECK line per item and exits 1 if any fails. Nothing is loaded into the FPGA:
it runs on the bitstream the card has (r8: CFG_HASH 93e29b18)."""
from __future__ import annotations

import argparse
import os
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if "--sim" in sys.argv:          # the board model as the board builds it (WIN 16 for WAITC, TMEM)
    os.environ.setdefault("OTPU_UARCH", "board")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from opentpu import isa as I  # noqa: E402
from opentpu.host.board import Board, check_hash, device_config  # noqa: E402
from opentpu.host.checks import PROG_AT, run_demo  # noqa: E402
from opentpu.host.opchecks import diag_image, mm_block_offset_case, op_checks  # noqa: E402
from opentpu.isasim import Machine  # noqa: E402

R8_HASH = 0x93e29b18
R8Q_HASH = 0xa905c3db       # r8 with GRQ QUAD (CAPS2 bit20, Config.TGQ4: C2)
# the area build (C4, r8_build_c4.cmd): C2's capabilities less TMM T5, BIN and BIN GSC, which no
# model on the card uses (OTPU_DROP_FEATURES; otpu_mxu HAS_T5 / HAS_BIN 0)
R8A_HASH = 0x37f51889
R8A_LEFT_OUT = ("tbin", "tbgsc", "td5")
# experiment A (r8_build_a.cmd): r8 less BIN / BIN GSC (T5 kept), r5's memory write path
R8X_HASH = 0x432e34c2
R8X_LEFT_OUT = ("tbin", "tbgsc")
# r5x lean (r8_build_lean.cmd): r5x less the TFN atan table too (otpu_se_comp HAS_ATAN 0: no model
# uses it), with the shared TFN ROM and one RAND generator (no capability change)
R8N_HASH = 0x852be9dc
R8N_LEFT_OUT = ("tbin", "tbgsc", "tatan")
# the builds with capabilities left out: CFG_HASH -> the CAPS2 bits it must not have
LEFT_OUT = {R8A_HASH: R8A_LEFT_OUT, R8X_HASH: R8X_LEFT_OUT, R8N_HASH: R8N_LEFT_OUT}
R8_CAPS2 = ("tgsc", "tbin", "tbgsc", "vcmp", "vu8", "tgrq", "vrand", "tcq", "tatan", "qst4",
            "ldx", "td5", "vbfly", "vaff", "qst6", "vcswp")
R8_SIM = ("TPAIR", "TQUAD", "TFN", "DSTEP", "TGSC", "VCMP", "TBIN", "TBGSC", "VU8", "TGRQ",
          "VRAND", "TCQ", "TATAN", "QST4", "LDX", "TD5", "VBFLY", "VAFF", "QST6", "VCSWP", "TGQ4")
IMG_MAX, DUMP, ZERO, END = 0x200000, 0x200000, 0x280000, 0x300000


def caps_checks(info: dict, sim: bool = False) -> list:
    """[(ok, what, detail)]: the bitstream's CAPS2 holds r8's capabilities (a build in LEFT_OUT:
    less its own, and those clear) and its CFG_HASH is one of r8's builds."""
    c2 = info.get("caps2") or {}
    h = (info.get("cfg") or {}).get("hash", 0)
    gone = LEFT_OUT.get(h, ())
    missing = [k for k in R8_CAPS2 if not c2.get(k) and k not in gone]
    out = [(not missing, "CAPS2: the r8 capabilities", "missing " + ", ".join(missing) if missing
            else "all of " + " ".join(k for k in R8_CAPS2 if c2.get(k))
            + (" (left out: " + " ".join(gone) + ")" if gone else ""))]
    if gone:
        extra = [k for k in gone if c2.get(k)]
        out.append((not extra, "CAPS2: the build leaves out " + " / ".join(gone),
                    ", ".join(extra)))
    known = (R8Q_HASH, *LEFT_OUT) if c2.get("tgq4") else (R8_HASH,)
    out.append((sim or h in known, "CFG_HASH is r8's",
                f"{h:#010x}, r8 {R8_HASH:#010x}, with GRQ QUAD {R8Q_HASH:#010x}, area build "
                f"{R8A_HASH:#010x}, experiment A / r5x {R8X_HASH:#010x}, r5x less atan "
                f"{R8N_HASH:#010x}"))
    return out


def area_refusals(b, gone=R8A_LEFT_OUT) -> list:
    """[(ok, what, detail)] on the area build (C4): the host refuses TMM T5 / BIN / BIN GSC
    programs (Board.load_program's CAPS2 checks, before anything is written to the card)."""
    from opentpu.host.board import ConfigMismatch
    head = [I.ld(0, 0, 128), I.qact(0, 1, 0, 1, 128)]
    progs = {"TMM T5": I.tmm(0x10000, 0x8000, 4000, 4, 5, 128, 4, 1, 0, 20, gsc=True,
                             dense=True),
             "TMM BIN": I.tmm(0x10000, 0x8000, 4000, 4, 8, 128, 4, 1, 0, 4, rsc=True,
                              binary=True),
             "TMM BIN GSC": I.tmm(0x10000, 0x8000, 4000, 4, 8, 128, 4, 1, 0, 32, gsc=True,
                                  binary=True)}
    if "td5" not in gone:
        progs.pop("TMM T5")
    if "tatan" in gone:
        progs["VOP atan"] = I.vop(I.V_ATAN, 0x100, 0, 0, 1, 128, 128, 128, 128)
    out = []
    for name, ins in progs.items():
        try:
            b.load(PROG_AT, head + [ins, I.halt()])
            out.append((False, f"the area build refuses {name}", "loaded"))
        except ConfigMismatch as e:
            out.append((True, f"the area build refuses {name}", str(e)[:80]))
    return out


def run_case(b: Board, card, prog: list, image: np.ndarray) -> tuple[bool, str]:
    """A test's program and DRAM image on the card and on the ISA simulator (the card's
    configuration): TMEM cleared first, all of it stored to DUMP last; DRAM [0, ZERO) compared."""
    used = np.flatnonzero(image)
    n = int(used[-1]) + 1 if used.size else 0
    if n > IMG_MAX:
        return False, f"image of {n} bytes above {IMG_MAX:#x}: not runnable here"
    T = card.TMEM_WORDS
    if 4 * T > ZERO - DUMP:
        T = (ZERO - DUMP) // 4
    body = list(prog)
    while body and body[-1].op == I.HALT and not body[-1].flags & I.F_CHAIN:
        body.pop()
    full = [I.ld(ZERO, 0, T)] + body + [I.st(DUMP, 0, T), I.halt()]
    ref = np.zeros(END, np.uint8)
    ref[:n] = image[:n]
    sl = Machine(dataclasses.replace(card, DRAM_BYTES=END), [full], [ref]).run().slices[0]
    b.write(ZERO, np.zeros(4 * T, np.uint8))
    b.write(0, ref[:ZERO])
    b.load(PROG_AT, full)
    t0 = time.time()
    st = b.run(timeout=60.0)
    got = b.read(0, ZERO)
    want = sl.dram[:ZERO]
    if st["instructions"][0] != sl.icount:
        return False, f"retired {st['instructions'][0]} of {sl.icount} instructions"
    bad = np.flatnonzero(got != want)
    if bad.size:
        a = int(bad[0]) // 4 * 4
        gw, ww = got.view("<u4"), want.view("<u4")
        where = "TMEM word %d" % ((a - DUMP) // 4) if a >= DUMP else "DRAM %#x" % a
        return False, (f"{bad.size} bytes differ, first at {where}: got {int(gw[a // 4]):#010x} "
                       f"want {int(ww[a // 4]):#010x}")
    return True, f"{st['cycles']} cycles, {time.time() - t0:.1f} s"


def r8_cases(card) -> list:
    """(name, program, image) of r8's multi-instruction cases, from the RTL tests' builders."""
    import test_cswp as CS
    import test_ldx as LX
    import test_qst4 as Q4
    import test_qst6 as Q6
    import test_tmm_cq as CQ
    import test_tmm_grq as GQ
    import test_tmm_t5 as T5
    import test_vrand as VR
    from opentpu.kernels import sort as SO
    prog, img = mm_block_offset_case(card)
    out = [("MM k0 scale offsets", prog, img)]
    # CQ1 is BIN's layout (TMM BIN CQ: Config.TBIN; the area build C4 leaves it out)
    for bits in ((1, 2, 3, 4) if getattr(card, "TBIN", False) else (2, 3, 4)):
        cfg, img, prog, _ = CQ.case(bits=bits, kb=3, m=min(4, card.MCOLS), n=12)
        out.append((f"TMM CQ{bits} (LUTSET, GSC epilogue)", prog, img))
    cfg, img, prog, _ = CQ.case(bits=4, kb=2, m=2, n=8, acc=True)
    out.append(("TMM CQ4 ACC", prog, img))
    for grq in (False, True):
        cfg, img, head, gsc, gq, _ = GQ.case(kb=5, m=3, n=12, mcols=min(4, card.MCOLS))
        out.append((f"TMM GSC{' GRQ' if grq else ''}", head + [gq if grq else gsc, I.halt()], img))
    if getattr(card, "TGQ4", False):             # GRQ QUAD: between a GRQ and itself, ACC
        for kb, acc, n in ((5, False, 20), (7, True, 36)):
            cfg, img, head, gq, quad = GQ.gq4_case(kb, acc, n, mcols=min(8, card.MCOLS))
            out.append((f"TMM GRQ QUAD (KB {kb}{', ACC' if acc else ''})",
                        head + [quad, gq, quad, I.halt()], img))
    for dense in ((False, True) if getattr(card, "TD5", False) else (False,)):
        _, (cfg, prog, img) = T5.run(24, 2, 3, gsc=True, dense=dense)
        out.append((f"TMM {'T5 (dense trits)' if dense else 'GSC 2-bit'}", prog, img))
    cfg, img, prog, _ = Q4.case(rows=9, kb=1, m=2)
    out.append(("QST Q4 (4-bit K) + TMM", prog, img))
    cfg, img, prog, _ = Q6.case(n=12, kb=2, m=3)
    out.append(("QST Q6 (6-bit K, three planes) + TMM CQ2 GRQ scores", prog, img))
    cfg, img, prog, _ = Q6.vcase(n=6, m=2, appends=(6, 7, 8, 9))
    out.append(("QST Q6T (6-bit V^T, appends) + P.V", prog, img))
    cfg, img, prog, _ = Q6.vcase(n=130, m=2)
    out.append(("QST Q6T 130 tokens (two D-blocks)", prog, img))
    for rows, n, desc in ((2, 64, True), (5, 256, False)):
        x = CS.sample(rows, n, rows + n)
        prog, _ = SO.bitonic(CS.A_T, CS.B_T, rows, n, n, descending=desc)
        cfg, img, full = CS.run(prog, x)
        out.append((f"VOP CSWP bitonic sort {rows}x{n}{' desc' if desc else ''}", full, img))
    c = VR.cfg()
    for rows, cols, drs, ars in ((1, 37, 37, 37), (4, 64, 64, 64)):
        n = (rows - 1) * ars + cols
        img, _ = VR.image(c, n, 1000)
        out.append((f"VOP RAND {rows}x{cols}", VR.program(c, n, rows, cols, drs, ars, 77), img))
    for raw in (False, True):
        img, prog, _ = LX.case(LX.cfg_d(card.D), 13, 200, 50, raw=raw)
        out.append((f"LDX{' raw' if raw else ''} 13 rows of 200", prog, img))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--sim", action="store_true", help="the board model configured as r8")
    ap.add_argument("--skip-ops", action="store_true", help="only the identity and r8 cases")
    ap.add_argument("--only", default="", help="only the r8 cases whose name holds this")
    a = ap.parse_args()
    if a.sim:
        from opentpu.host.board import SimTransport
        # r8's build (r8_build.cmd: bonsai2_validate.configuration() + the r8 flags)
        params = {k: 1 for k in R8_SIM}
        params.update(MCOLS=8, MXU_E=8, TMEM_WORDS=131072, SEQ_WC=1, WIN=16)   # WAITC: WIN <= 16 (the board)
        b = Board(SimTransport(ch_bytes=1 << 22, stall=30, seed=7, params=params))
    else:
        from opentpu.host.board import XdmaTransport
        b = Board(XdmaTransport(a.dev))
    fails = 0

    def check(ok, what, detail=""):
        nonlocal fails
        print(f"CHECK {'PASS' if ok else 'FAIL'}: {what}" + (f" ({detail})" if detail else ""),
              flush=True)
        fails += 0 if ok else 1

    info = b.info()
    c2 = info.get("caps2") or {}
    print("build", hex(info.get("build_id") or 0), "core_khz", info.get("core_khz"),
          "cfg hash", hex((info.get("cfg") or {}).get("hash", 0)), flush=True)
    for ok, what, detail in caps_checks(info, a.sim):
        check(ok, what, detail)
    if (info.get("cfg") or {}).get("hash", 0) in LEFT_OUT:
        for ok, what, detail in area_refusals(b, LEFT_OUT[info["cfg"]["hash"]]):
            check(ok, what, detail)
    try:
        check_hash(info)
        check(True, "CFG_HASH matches this host's Config")
    except Exception as e:                      # noqa: BLE001
        check(False, "CFG_HASH matches this host's Config", str(e))
    card = device_config(info)
    if not a.skip_ops and not a.only:
        img = diag_image(card)
        caps = info.get("caps") or {}
        for group, name, prog in op_checks(card, gen=bool(caps.get("gen")),
                                           waitw=bool(caps.get("waitw")),
                                           gather=bool(c2.get("gather"))):
            try:
                ok, msg, _ = run_demo(b, card, prog, img)
            except Exception as e:              # noqa: BLE001
                ok, msg = False, f"{type(e).__name__}: {e}"
            check(ok, f"op {group}: {name}", msg)
    cases = [case for case in r8_cases(card) if a.only in case[0]]
    if not cases:
        check(False, "r8 case selection", f"no case matched {a.only!r}")
    for name, prog, img in cases:
        try:
            ok, msg = run_case(b, card, prog, img)
        except Exception as e:                  # noqa: BLE001
            ok, msg = False, f"{type(e).__name__}: {e}"
        check(ok, f"r8 {name}", msg)
    print(f"== {fails} failed", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
