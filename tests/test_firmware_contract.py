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
    assert contract["firmware"]["version"] == {"D": 128, "MCOLS": 8, "LANES": 8}


def test_current_firmware_is_r5x():
    """The current firmware is r5x (c42beb65): the top-level entry and its "known" entry agree, and
    r8_build.cmd's CFG_HASH check gives its hash with the build's defaults (no BIN / BIN GSC)."""
    import os
    import subprocess
    import sys

    contract = firmware_contract.load_contract(ROOT / "firmware_contract.json")
    fw = contract["firmware"]
    assert fw["build_id"] == "0xc42beb65" and fw["source_commit"].startswith("c42beb65")
    current = [k for k in contract["known"] if k.get("status") == "current"]
    assert len(current) == 1
    k = current[0]
    assert (k["build_id"], k["cfg_hash"], k["bitstream_sha256"], k["version"]) == (
        fw["build_id"], fw["cfg_hash"], fw["bitstream_sha256"], fw["version"])
    info = {"build_id": 0xc42beb65, "cfg": {"hash": 0x432e34c2}, "D": 128, "MCOLS": 8, "LANES": 8}
    assert firmware_contract._match_known(contract, info)["status"] == "current"
    line = next(l for l in (ROOT / "tools/nextgen/windows/r8_build.cmd").read_text().splitlines()
                if "CFG_HASH check" in l and l.startswith("python -c"))
    code = line[len('python -c "'):-1]
    env = {k: v for k, v in os.environ.items() if k != "OTPU_DROP_FEATURES"}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                         text=True, check=True).stdout
    assert f"CFG_HASH check {int(fw['cfg_hash'], 0):08x}" in out
    env["OTPU_DROP_FEATURES"] = "none"          # every capability: r8's C2 / C3 hash
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                         text=True, check=True).stdout
    assert "CFG_HASH check a905c3db" in out
    ps = (ROOT / "tools/build_r8_133.ps1").read_text()
    for knob, value in (("OTPU_MEMCH_WREG", "'0'"), ("OTPU_LITEDRAM_RTW", "'3'"),
                        ("TRACE_DEPTH", "'16384'"), ("OTPU_DROP_FEATURES", "'TBIN TBGSC'")):
        assert f"{knob} = {value}" in ps


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
