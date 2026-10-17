import importlib.util
import hashlib
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
    assert contract["firmware"]["version"] == {"D": 128, "MCOLS": 4, "LANES": 8}


def test_rejects_unknown_contract_schema(tmp_path):
    manifest = tmp_path / "contract.json"
    manifest.write_text('{"schema_version": 2, "firmware": {}, "host": {}}')

    with pytest.raises(firmware_contract.ContractMismatch, match="unsupported"):
        firmware_contract.load_contract(manifest)


def test_bitstream_report_requires_manifest_hash(tmp_path):
    image = tmp_path / "otpu.bit"
    image.write_bytes(b"verified bitstream")
    contract = {"firmware": {"bitstream_sha256": hashlib.sha256(image.read_bytes()).hexdigest().upper()}}

    assert firmware_contract.bitstream_report(contract, image)["path"] == str(image)

    contract["firmware"]["bitstream_sha256"] = "0" * 64
    with pytest.raises(firmware_contract.ContractMismatch, match="do not load"):
        firmware_contract.bitstream_report(contract, image)
