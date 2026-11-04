import csv
import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "nextgen" / "ingest_reports.py"
SPEC = importlib.util.spec_from_file_location("nextgen_ingest_reports", MODULE_PATH)
assert SPEC and SPEC.loader
INGEST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INGEST)


def test_parse_vivado_utilization_rows():
    report = """\
| Slice LUTs                 | 12,345 | 5,760 | 214.32 |
| Slice Registers            | 8,901  | 1,152 | 772.66 |
| DSPs                       | 128    | 600   | 21.33  |
| Block RAM Tile             | 64     | 600   | 10.67  |
| URAM                       | 0      | 0     | 0.00   |
"""
    assert INGEST.parse_utilization(report) == {
        "LUT": 12345, "FF": 8901, "DSP": 128, "BRAM_TILE": 64, "URAM": 0,
    }


def test_parse_vivado_timing_header_and_data():
    report = """\
Design Timing Summary
WNS(ns) TNS(ns) TNS Failing Endpoints
------- ------- ---------------------
 0.032   0.000   0
"""
    assert INGEST.parse_timing_summary(report) == {"WNS_ns": 0.032, "TNS_ns": 0.0}


def test_report_parsers_fail_explicitly_when_data_is_missing():
    with pytest.raises(ValueError, match="utilization"):
        INGEST.parse_utilization("No matching report table")
    with pytest.raises(ValueError, match="timing-summary"):
        INGEST.parse_timing_summary("WNS and TNS were not reported")


def test_latency_csv_schema_and_rows(tmp_path):
    report = tmp_path / "latency.csv"
    report.write_text("workload,metric,value,unit\nqwen,decode,12.5,tok/s\n", encoding="utf-8")
    assert INGEST.parse_latency_csv(report) == [
        {"workload": "qwen", "metric": "decode", "value": "12.5", "unit": "tok/s"},
    ]
    report.write_text("model,latency\nqwen,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        INGEST.parse_latency_csv(report)


def test_build_record_preserves_provenance_and_unverified_status(tmp_path):
    util = tmp_path / "util.rpt"
    timing = tmp_path / "timing.rpt"
    latency = tmp_path / "latency.csv"
    util.write_text("| DSPs | 10 | 20 | 50 |\n", encoding="utf-8")
    timing.write_text("WNS(ns) TNS(ns)\n------- -------\n0.1 0.0\n", encoding="utf-8")
    with latency.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["workload", "metric", "value", "unit"])
        writer.writeheader()
        writer.writerow({"workload": "test", "metric": "latency", "value": "5", "unit": "ms"})
    record = INGEST.build_record(util, timing, latency, "abc123", "xc7k480tffg1156-2")
    assert record["evidence_class"] == "user_supplied_report_ingest"
    assert record["commit"] == "abc123"
    assert record["utilization"]["DSP"] == 10
    assert "does not run Vivado" in record["qualification"]
    assert json.loads(json.dumps(record))["timing"]["WNS_ns"] == 0.1
