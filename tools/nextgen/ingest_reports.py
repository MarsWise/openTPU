"""Ingest user-supplied Vivado text reports and benchmark CSV into a provenance JSON record."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys


_RESOURCE_NAMES = {
    "slice luts": "LUT",
    "clb luts": "LUT",
    "slice registers": "FF",
    "clb registers": "FF",
    "dsps": "DSP",
    "dsp48e1s": "DSP",
    "block ram tile": "BRAM_TILE",
    "ramb36/fifo36": "RAMB36",
    "ramb18/fifo18": "RAMB18",
    "uram": "URAM",
}
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


def parse_utilization(text: str) -> dict[str, int]:
    """Read recognized resource rows from a Vivado utilization report table."""
    resources = {}
    for line in text.splitlines():
        match = re.match(r"^\s*\|\s*([^|]+?)\s*\|\s*([\d,]+)\s*\|", line)
        if match:
            name = " ".join(match.group(1).lower().split())
            key = _RESOURCE_NAMES.get(name)
            if key:
                resources[key] = int(match.group(2).replace(",", ""))
    if not resources:
        raise ValueError("no recognized Vivado utilization rows found")
    return resources


def parse_timing_summary(text: str) -> dict[str, float]:
    """Read WNS/TNS from the first data row after a Vivado timing-summary header."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if "WNS(ns)" not in line or "TNS(ns)" not in line:
            continue
        for row in lines[index + 1:index + 6]:
            values = _NUMBER.findall(row)
            if len(values) >= 2:
                return {"WNS_ns": float(values[0]), "TNS_ns": float(values[1])}
    raise ValueError("Vivado timing-summary WNS(ns)/TNS(ns) row not found")


def parse_latency_csv(path: Path) -> list[dict[str, str]]:
    """Read benchmark rows with workload, metric, value and unit columns."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"workload", "metric", "value", "unit"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("latency CSV must contain workload, metric, value, and unit columns")
        rows = []
        for row_number, row in enumerate(reader, start=2):
            if not row["workload"] or not row["metric"] or not row["unit"]:
                raise ValueError(f"latency CSV row {row_number} has an empty required field")
            try:
                float(row["value"])
            except ValueError as exc:
                raise ValueError(f"latency CSV row {row_number} has a non-numeric value") from exc
            rows.append(dict(row))
    if not rows:
        raise ValueError("latency CSV has no result rows")
    return rows


def build_record(util_path: Path, timing_path: Path, latency_path: Path,
                 commit: str, part: str) -> dict:
    """Create a provenance record without upgrading supplied reports to independently verified."""
    return {
        "evidence_class": "user_supplied_report_ingest",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "part": part,
        "utilization": parse_utilization(util_path.read_text(encoding="utf-8-sig")),
        "timing": parse_timing_summary(timing_path.read_text(encoding="utf-8-sig")),
        "latency": parse_latency_csv(latency_path),
        "sources": {
            "utilization": str(util_path),
            "timing": str(timing_path),
            "latency": str(latency_path),
        },
        "qualification": (
            "Values were parsed from user-supplied files; this tool does not run Vivado, "
            "verify report authenticity, or validate a physical implementation."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utilization", type=Path, required=True, help="Vivado utilization report")
    parser.add_argument("--timing", type=Path, required=True, help="Vivado timing summary report")
    parser.add_argument("--latency", type=Path, required=True, help="benchmark CSV")
    parser.add_argument("--commit", required=True, help="source commit SHA")
    parser.add_argument("--part", required=True, help="FPGA part identifier")
    parser.add_argument("--output", type=Path, required=True, help="output JSON file")
    args = parser.parse_args(argv)

    for report in (args.utilization, args.timing, args.latency):
        if not report.is_file():
            parser.error(f"report does not exist or is not a file: {report}")
    try:
        record = build_record(args.utilization, args.timing, args.latency, args.commit, args.part)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))

    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
