import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "firmware_contract", ROOT / "tools" / "firmware_contract.py"
)
firmware_contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(firmware_contract)


def test_host_contract_matches_tmm_instruction():
    contract = firmware_contract.load_contract(ROOT / "firmware_contract.json")

    report = firmware_contract.host_report(contract)

    assert report["tmm_opcode"] == "0x23"
    assert len(report["source_commit"]) == 40


def test_rejects_unknown_contract_schema(tmp_path):
    manifest = tmp_path / "contract.json"
    manifest.write_text('{"schema_version": 2, "firmware": {}, "host": {}}')

    with pytest.raises(firmware_contract.ContractMismatch, match="unsupported"):
        firmware_contract.load_contract(manifest)
