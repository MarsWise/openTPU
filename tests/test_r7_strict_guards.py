"""Reject a strict claim before a hybrid model or physical device can start."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('script,args,reason', [
    ('diffusion_run.py', ['--tiny'], 'GroupNorm'),
    ('yolox_graph.py', ['unused-repo', 'unused-weights'], 'letterboxes'),
])
def test_hybrid_runner_rejects_strict_before_opening_model_or_board(script, args, reason):
    cmd = [sys.executable, str(ROOT / 'tools' / script), *args,
           '--backend', 'board', '--strict-fpga-only']
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'strict FPGA-only execution is unavailable' in result.stderr
    assert reason in result.stderr


@pytest.mark.skipif(os.name == 'nt', reason='the chat host runtime requires POSIX fcntl')
def test_chat_rejects_strict_before_loading_checkpoint_or_board():
    cmd = [sys.executable, '-m', 'opentpu.host.chat', '--model', 'unused',
           '--backend', 'board', '--strict-fpga-only']
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'strict FPGA-only execution is unavailable' in result.stderr
    assert 'prefill token' in result.stderr
