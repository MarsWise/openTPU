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


def test_known_bitstreams_match_on_build_hash_and_version():
    """r4, r5-m8 (2f2320e2) and the pending r5-m4 are known: the card's build, CFG_HASH and VERSION pick one."""
    contract = firmware_contract.load_contract(ROOT / "firmware_contract.json")
    r4 = {"build_id": 0xbb27096f, "cfg": {"hash": 0x3fb1359e}, "D": 128, "MCOLS": 4, "LANES": 8}
    assert firmware_contract._match_known(contract, r4)["name"].startswith("r4")
    m8 = {"build_id": 0x2f2320e2, "cfg": {"hash": 0x3b7f244e}, "D": 128, "MCOLS": 8, "LANES": 8}
    assert firmware_contract._match_known(contract, m8)["name"].startswith("r5-m8")
    # the built r5-m8 is pinned: another m8 build with its hash is not it
    assert firmware_contract._match_known(contract, dict(m8, build_id=0x12345678)) is None
    m4 = {"build_id": 0x12345678, "cfg": {"hash": 0x603d5a6f}, "D": 128, "MCOLS": 4, "LANES": 8}
    assert firmware_contract._match_known(contract, m4)["name"].startswith("r5-m4")   # pending
    bad = dict(r4, cfg={"hash": 0x4ffd1e17})
    assert firmware_contract._match_known(contract, bad) is None
