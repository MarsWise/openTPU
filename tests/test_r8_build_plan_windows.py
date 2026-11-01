"""Check the Windows build recipe's capability and architecture identity."""
import os
import re
import subprocess
from pathlib import Path

import pytest

from opentpu.target import arch_hash
from tools.bonsai2_validate import configuration
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.name != "nt", reason="PowerShell build recipe is Windows-specific")
@pytest.mark.parametrize("script,enabled", [("build_r7_133.ps1", False), ("build_r8_133.ps1", True)])
def test_bfly_plan_matches_architecture(script, enabled):
    result = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                             str(ROOT / "tools" / script), "-Mode", "Build", "-Plan"],
                            cwd=ROOT, text=True, capture_output=True, check=True)
    text = result.stdout
    assert re.search(r"OTPU_VBFLY_GEN\s+" + str(int(enabled)) + r"\b", text)
    if enabled:
        cfg = replace(configuration(), TGSC=True, VCMP=True, TBIN=True, TBGSC=True,
                      VU8=True, TGRQ=True, VRAND=True, TCQ=True, TATAN=True,
                      QST4=True, LDX=True, TD5=True, VBFLY=True)
        assert f"hash={arch_hash(cfg):08x}" in text
