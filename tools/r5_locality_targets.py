"""Select placement experiments from r5_path_analysis.tcl's endpoint CSV (stdlib only).

The TSV is data, never sourced as Tcl. Coordinates remain reference sites; Vivado
resolves the corresponding SLICE windows on the actual device at replay time.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

FIELDS = {
    "slack", "from_clock", "to_clock", "start", "end", "start_site", "end_site",
}
RMW = re.compile(r"^(.*?/u_mxu/g_rmw\[(\d+)\]\.u_y/u_m)/")
TMEM = re.compile(r"^(.*?/u_tmem/.*?\.u_pb)/evr_reg(?:_rep.*)?$")
SITE = re.compile(r"^(?:DSP48|RAMB18|RAMB36)_X\d+Y\d+$")
TSV_FIELDS = ("id", "kind", "prefix", "anchor", "site", "worst_ns", "endpoints")


def select_targets(path: Path, scope: str = "both") -> tuple[list[dict], dict]:
    groups: dict[str, dict] = {}
    pairs: Counter = Counter()
    count = 0
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not FIELDS.issubset(reader.fieldnames or []):
            raise ValueError("paths.csv is missing required r5_path_analysis columns")
        for row in reader:
            if None in row or any(row[k] is None for k in FIELDS):
                raise ValueError("Malformed endpoint CSV row")
            slack = float(row["slack"])
            if not math.isfinite(slack):
                raise ValueError("Non-finite endpoint slack")
            if slack >= 0:
                continue
            count += 1
            pairs[f'{row["from_clock"]}->{row["to_clock"]}'] += 1
            candidates = []
            if scope in ("both", "rmw") and row["from_clock"] == row["to_clock"] == "core_clk":
                # Both ends matter: some failing paths feed the multiplier's DSP.
                for cell in (row["start"], row["end"]):
                    m = RMW.match(cell)
                    if m:
                        candidates.append((f"rmw_{m[2]}", "rmw", m[1]))
            if scope in ("both", "tmem") and row["from_clock"] == row["to_clock"] == "core_clk2x":
                m = TMEM.match(row["start"])
                if m and row["end"].startswith(m[1] + "/"):
                    # A shared/merged phase register must not be assigned to two boxes.
                    candidates.append(("tmem_" + re.sub(r"\D+", "_", m[1].split("/u_tmem/", 1)[1]).strip("_"),
                                       "tmem", m[1]))
            for ident, kind, prefix in set(candidates):
                group = groups.setdefault(ident, dict(id=ident, kind=kind, prefix=prefix,
                                                     worst_ns=slack, endpoints=0, anchors={}))
                if group["prefix"] != prefix:
                    raise ValueError(f"Ambiguous target id {ident}")
                group["worst_ns"] = min(slack, group["worst_ns"])
                group["endpoints"] += 1
                for cell, site in ((row["start"], row["start_site"]), (row["end"], row["end_site"])):
                    expected = "DSP48" if kind == "rmw" else "RAMB"
                    if cell.startswith(prefix + "/") and site.startswith(expected):
                        if not SITE.fullmatch(site):
                            raise ValueError(f"Invalid reference site {site}")
                        previous = group["anchors"].setdefault(cell, site)
                        if previous != site:
                            raise ValueError(f"Inconsistent reference location for {cell}")
    targets = sorted(groups.values(), key=lambda g: (g["kind"], g["id"]))
    for target in targets:
        if not target["anchors"]:
            raise ValueError(f'No DSP/BRAM anchor was observed for {target["prefix"]}')
    if scope != "none" and not targets:
        raise ValueError(f"No targets for scope {scope}")
    return targets, dict(failing_endpoint_rows=count, clock_pairs=dict(sorted(pairs.items())),
                         target_groups=len(targets), target_endpoint_rows=sum(t["endpoints"] for t in targets),
                         targets=targets)


def write_targets(targets: list[dict], output: Path) -> None:
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE)
        writer.writerow(TSV_FIELDS)
        for group in targets:
            for cell, site in sorted(group["anchors"].items()):
                values = [group["id"], group["kind"], group["prefix"], cell, site,
                          f'{group["worst_ns"]:.3f}', group["endpoints"]]
                if any(any(c in str(v) for c in "\t\r\n\"") for v in values):
                    raise ValueError("Unsupported control/quote character in target data")
                writer.writerow(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--scope", choices=("both", "rmw", "tmem", "none"), default="both")
    args = parser.parse_args()
    targets, summary = select_targets(args.paths, args.scope)
    write_targets(targets, args.output)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f'{summary["failing_endpoint_rows"]} failing endpoint rows; {len(targets)} locality groups')


if __name__ == "__main__":
    main()
