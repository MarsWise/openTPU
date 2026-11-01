from types import SimpleNamespace

import numpy as np
import pytest

from opentpu.llm.ptq1 import ptq1_encode
from tools import bonsai_ptq1_check as check


@pytest.mark.parametrize("problem", ["missing", "type", "shape", "short", "both_short"])
def test_comparison_fails_closed_for_bad_second_tensor(monkeypatch, problem):
    ptq = ptq1_encode(np.ones((1, 128), dtype=np.uint8), np.ones(1, np.float32))
    pq = np.r_[np.frombuffer(np.float16(1).tobytes(), np.uint8), np.full(32, 0x55, np.uint8)]
    def tensor(typ, shape=(128, 1), nbytes=28):
        return SimpleNamespace(type=typ, shape=shape, nbytes=nbytes)
    class File:
        def __init__(self, path):
            self.path = path
            self.tensors = {"good": tensor(check.PTQ1_0), "bad": tensor(check.PTQ1_0)} if path == "a" else {
                "good": tensor(check.PQ2_0, nbytes=34),
                "bad": tensor(check.PQ2_0, (64, 2) if problem == "shape" else (128, 1), 34)}
            if problem == "both_short":
                self.tensors["bad"].shape = (256, 1)
            if path == "b" and problem == "missing":
                del self.tensors["bad"]
            if path == "b" and problem == "type":
                self.tensors["bad"].type = -1
        def raw(self, name):
            return ptq if self.path == "a" else pq[:-1] if name == "bad" and problem == "short" else pq
    monkeypatch.setattr(check, "GGUF", File)
    assert check.main(["a", "b"]) == 1
