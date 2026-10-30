"""The hybrid split runner must not silently claim strict FPGA-only execution."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def invoke(*args):
    return subprocess.run([sys.executable, str(ROOT / 'tools' / 'smolvla_split.py'), *args],
                          cwd=ROOT, capture_output=True, text=True)


def test_help_needs_no_model_or_board():
    result = invoke('--help')
    assert result.returncode == 0
    assert 'hybrid runner' in result.stdout


def test_strict_mode_fails_before_loading_model_or_board():
    result = invoke('--mode', 'board', '--ref', 'unused', '--strict-fpga-only')
    assert result.returncode != 0
    assert 'not implemented by this hybrid' in result.stderr


def test_invalid_accuracy_threshold_fails_before_loading_model():
    result = invoke('--mode', 'board', '--ref', 'unused', '--max-rel-l2', '-1')
    assert result.returncode != 0
    assert 'finite and nonnegative' in result.stderr
