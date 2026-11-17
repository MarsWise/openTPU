"""Compare two Vivado `report_utilization -hierarchical` reports (a build's reports/util_hier.rpt):
the whole design and the units where r8's area sits, with the difference B - A.

    python tools/util_compare.py <A util_hier.rpt or build dir> <B ...> [--all]

A build directory means its reports/util_hier.rpt. The default rows: the top, u_bd (XDMA), the
slice and its units (u_mxu with its own logic "(u_mxu)", which holds the weight decode and the
CQ LUT RAMs, u_tmm_gsc, u_vpu, u_tmem, u_quant, u_seq, u_dma), the controller and the trace;
--all prints every instance present in both. Numbers are Vivado's (placed / routed: the report's
Design State); the evidence class is SYNTHESIS ESTIMATED for a routed design."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

COLS = ("lut", "logic", "lutram", "srl", "ff", "ramb36", "ramb18", "dsp")
SHOW = ("otpu_fpga_top_ld", "u_bd", "u_slice", "u_mxu", "(u_mxu)", "u_tmm_gsc", "u_vpu", "u_tmem",
        "u_quant", "u_seq", "u_dma", "u_act", "u_ctrl", "g_trace", "u_trace")


def parse(path: Path) -> dict:
    """{dotted instance path: {column: int}, ...}; '_state': the report's Design State."""
    text = path.read_text(errors="replace")
    out, stack = {}, []
    m = re.search(r"\|\s*Design State\s*:\s*(\S+)", text)
    out["_state"] = m.group(1) if m else "?"
    for line in text.splitlines():
        if not line.startswith("|") or "Instance" in line:
            continue
        cells = line.split("|")[1:-1]
        if len(cells) != 2 + len(COLS):
            continue
        raw = cells[0]
        name = raw.strip()
        if not name:
            continue
        try:
            vals = [int(c.strip()) for c in cells[2:]]
        except ValueError:
            continue
        depth = (len(raw) - len(raw.lstrip(" ")) - 1) // 2
        del stack[depth:]
        stack.append(name)
        out[".".join(stack)] = dict(zip(COLS, vals), name=name, depth=depth)
    return out


def report_of(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir():
        p = p / "reports" / "util_hier.rpt"
    if not p.is_file():
        raise SystemExit(f"{p}: no util_hier.rpt")
    return p


def rows(a: dict, b: dict, every: bool) -> list:
    keys = [k for k in a if not k.startswith("_") and k in b]
    if every:
        return keys
    return [k for k in keys if a[k]["name"] in SHOW and a[k]["depth"] <= 5]


def table(a: dict, b: dict, keys: list, la: str, lb: str) -> str:
    w = max([len(k) for k in keys] + [8])
    head = (f"{'instance':<{w}} " + " ".join(f"{c:>15}" for c in ("LUT", "LUTRAM", "FF", "RAMB36",
                                                                    "DSP")))
    lines = [f"A {la} ({a['_state']})   B {lb} ({b['_state']})   cells: A -> B (B - A)", head]
    for k in keys:
        cells = []
        for c in ("lut", "lutram", "ff", "ramb36", "dsp"):
            x, y = a[k][c], b[k][c]
            cells.append(f"{x:>6}->{y:<6}" + (f"{y - x:+d}" if y != x else ""))
        lines.append(f"{k:<{w}} " + " ".join(f"{s:>15}" for s in cells))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--all", action="store_true", help="every instance in both reports")
    a = ap.parse_args(argv)
    pa, pb = report_of(a.a), report_of(a.b)
    ra, rb = parse(pa), parse(pb)
    keys = rows(ra, rb, a.all)
    if not keys:
        print("no common instances", file=sys.stderr)
        return 1
    def label(arg, p):
        return Path(arg).name if Path(arg).is_dir() else p.stem
    print(table(ra, rb, keys, label(a.a, pa), label(a.b, pb)))
    top = next(k for k in keys if "." not in k)
    d = rb[top]["lut"] - ra[top]["lut"]
    print(f"\ndesign LUTs {ra[top]['lut']} -> {rb[top]['lut']} ({d:+d}, "
          f"{100 * d / 298600:+.2f} % of the XC7K480T's 298 600)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
