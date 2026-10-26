#!/usr/bin/env python3
"""Verify that a host checkout is paired with the expected openTPU firmware."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "firmware_contract.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ContractMismatch(RuntimeError):
    """The host checkout or card does not meet the firmware contract."""


def load_contract(path: Path) -> dict:
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContractMismatch(f"cannot read firmware contract {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ContractMismatch(f"invalid firmware contract {path}: {exc}") from exc
    if contract.get("schema_version") != 1:
        raise ContractMismatch("unsupported firmware contract schema")
    if not isinstance(contract.get("firmware"), dict) or not isinstance(contract.get("host"), dict):
        raise ContractMismatch("firmware contract must contain firmware and host objects")
    return contract


def host_report(contract: dict) -> dict:
    from opentpu import isa

    expected_tmm = int(contract["host"]["tmm_opcode"], 0)
    actual_tmm = getattr(isa, "TMM", None)
    if actual_tmm != expected_tmm:
        actual = "missing" if actual_tmm is None else f"{actual_tmm:#x}"
        raise ContractMismatch(
            f"host TMM opcode is {actual}, expected {expected_tmm:#x}; "
            "check out the firmware contract's feature branch"
        )
    return {
        "source_commit": contract["firmware"]["source_commit"],
        "tmm_opcode": f"{actual_tmm:#x}",
    }


def _match_known(contract: dict, info: dict) -> dict | None:
    """The contract's "known" bitstream (r3, r4, r5-m4, r5-m8: each its build, CFG_HASH and
    VERSION) that the card's registers name, or None. A pending entry (build_id "pending",
    before its build) matches on CFG_HASH and VERSION."""
    cfg_hash = (info.get("cfg") or {}).get("hash")
    for k in contract.get("known", []):
        bid = k.get("build_id", "pending")
        if bid != "pending" and info.get("build_id") != int(bid, 0):
            continue
        if cfg_hash is not None and int(k.get("cfg_hash", "0"), 0) != cfg_hash:
            continue
        if any(info.get(key) != val for key, val in k["version"].items()):
            continue
        return {**k, "build_id": bid if bid != "pending" else f"{info.get('build_id') or 0:#x}"}
    return None


def card_report(contract: dict, dev: str) -> dict:
    from opentpu.host.board import Board, XdmaTransport

    transport = XdmaTransport(dev, dma=False)
    try:
        board = Board(transport, lock=False, calibrate=False)
        info = board.info()
    finally:
        transport.close()

    expected = _match_known(contract, info) or contract["firmware"]
    expected_version = expected["version"]
    actual_version = {key: info[key] for key in expected_version}
    mismatches = [
        f"{key}={actual_version[key]} (expected {value})"
        for key, value in expected_version.items()
        if actual_version[key] != value
    ]
    expected_build_id = int(expected["build_id"], 0)
    if info["build_id"] != expected_build_id:
        actual = "unavailable" if info["build_id"] is None else f"{info['build_id']:#010x}"
        mismatches.append(f"build_id={actual} (expected {expected_build_id:#010x})")
    if not (info.get("caps") or {}).get("mx"):
        mismatches.append("CAP_MX is clear")
    if mismatches:
        raise ContractMismatch(
            "loaded firmware does not match this host contract: " + "; ".join(mismatches)
        )

    return {
        "device": dev,
        "build_id": f"{info['build_id']:#010x}",
        "version": actual_version,
        "cfg_hash": None if not info["cfg"] else f"{info['cfg']['hash']:#010x}",
        "bitstream_sha256": expected["bitstream_sha256"],
        "name": expected.get("name", contract["firmware"].get("name")),
        "part": expected.get("part", contract["firmware"]["part"]),
    }


def bitstream_report(contract: dict, path: Path) -> dict:
    try:
        with path.open("rb") as image:
            hasher = hashlib.sha256()
            for block in iter(lambda: image.read(1 << 20), b""):
                hasher.update(block)
            digest = hasher.hexdigest().upper()
    except OSError as exc:
        raise ContractMismatch(f"cannot hash bitstream {path}: {exc}") from exc
    expected = contract["firmware"]["bitstream_sha256"]
    if digest != expected:
        raise ContractMismatch(
            f"bitstream SHA-256 is {digest}, expected {expected}: do not load this image"
        )
    return {"path": str(path), "sha256": digest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="check the openTPU BitNET host/firmware compatibility contract"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dev", default="/dev/xdma0", help="XDMA device prefix")
    parser.add_argument("--bitstream", type=Path, help="verify this .bit file before loading it")
    parser.add_argument("--card", action="store_true", help="also read the card's BAR0 registers")
    parser.add_argument("--json", action="store_true", help="write the successful report as JSON")
    args = parser.parse_args(argv)

    try:
        contract = load_contract(args.manifest)
        report = {"host": host_report(contract)}
        if args.bitstream:
            report["bitstream"] = bitstream_report(contract, args.bitstream)
        if args.card:
            report["card"] = card_report(contract, args.dev)
    except ContractMismatch as exc:
        print(f"firmware-contract: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            f"firmware-contract: host TMM {report['host']['tmm_opcode']} matches "
            f"{report['host']['source_commit'][:12]}"
        )
        if "card" in report:
            card = report["card"]
            print(
                f"firmware-contract: {card['device']} build {card['build_id']}, "
                f"D={card['version']['D']} MCOLS={card['version']['MCOLS']} "
                f"LANES={card['version']['LANES']}"
            )
        if "bitstream" in report:
            print(f"firmware-contract: verified {report['bitstream']['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
