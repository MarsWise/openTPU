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
    assert re.search(r"OTPU_VAFF_GEN\s+" + str(int(enabled)) + r"\b", text)
    assert re.search(r"OTPU_QST6_GEN\s+" + str(int(enabled)) + r"\b", text)
    assert re.search(r"OTPU_VCSWP_GEN\s+" + str(int(enabled)) + r"\b", text)
    if enabled:
        cfg = replace(configuration(), TGSC=True, VCMP=True, TBIN=True, TBGSC=True,
                      VU8=True, TGRQ=True, VRAND=True, TCQ=True, TATAN=True,
                      QST4=True, LDX=True, TD5=True, VBFLY=True, VAFF=True, QST6=True, VCSWP=True)
        assert f"hash={arch_hash(cfg):08x}" in text


@pytest.mark.skipif(os.name != "nt", reason="Windows batch launcher")
def test_batch_plan_uses_its_own_checkout():
    script = ROOT / "tools" / "nextgen" / "windows" / "r8_build.cmd"
    result = subprocess.run(["cmd.exe", "/c", str(script), "Plan"], cwd=ROOT.parent,
                            input="", text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OTPU_VBFLY_GEN" in result.stdout and "OTPU_VAFF_GEN" in result.stdout
    assert str(ROOT).lower() in result.stdout.lower()


@pytest.mark.skipif(os.name != "nt", reason="Windows batch launcher")
def test_batch_launcher_propagates_powershell_failure(tmp_path, monkeypatch):
    from shutil import copyfile
    launcher = tmp_path / "tools" / "nextgen" / "windows" / "r8_build.cmd"
    launcher.parent.mkdir(parents=True)
    copyfile(ROOT / "tools" / "nextgen" / "windows" / "r8_build.cmd", launcher)
    (tmp_path / "rtl" / "shell").mkdir(parents=True)
    (tmp_path / "rtl" / "shell" / "otpu_ctrl.sv").write_text("VAFF\nQST6\nVCSWP\n")
    (tmp_path / "tools" / "build_r8_133.ps1").write_text("throw 'intentional failure'\n")
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    result = subprocess.run(["cmd.exe", "/c", str(launcher), "Plan"], cwd=tmp_path,
                            input="", text=True, capture_output=True, timeout=60)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "intentional failure" in result.stdout + result.stderr


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell build script")
def test_prepare_refuses_untracked_rtl_before_vivado(tmp_path):
    from shutil import copyfile
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    copyfile(ROOT / "tools" / "build_r5_timing.ps1", tools_dir / "build_r5_timing.ps1")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "tools/build_r5_timing.ps1"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-q", "-m", "baseline"], cwd=tmp_path, check=True)
    rtl = tmp_path / "rtl"
    rtl.mkdir()
    (rtl / "untracked.sv").write_text("module untracked; endmodule\n")
    vivado = tmp_path / "fake_vivado.bat"
    marker = tmp_path / "vivado_was_launched.txt"
    vivado.write_text(f'@echo launched > "{marker}"\n', encoding="ascii")
    result = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                             str(tools_dir / "build_r5_timing.ps1"), "-Mode", "Prepare",
                             "-Vivado", str(vivado)], cwd=tmp_path, capture_output=True, text=True,
                            timeout=60)
    assert result.returncode != 0
    assert "untracked" in (result.stdout + result.stderr).lower()
    assert not marker.exists(), "Vivado must not start with untracked RTL"
