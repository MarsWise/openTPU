"""tools/ddr_diag.py wr: a word stored wrong against a word read wrong."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("ddr_diag", Path(__file__).parents[1] / "tools" / "ddr_diag.py")
dd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dd)


class Mem:
    def __init__(self, n, flip_write=None, flip_read=None):
        self.m = np.zeros(n, np.uint8)
        self.fw, self.fr, self.reads = flip_write, flip_read, 0

    def mem_write(self, ch, off, data):
        self.m[off:off + len(data)] = data
        if self.fw is not None:
            self.m[off + self.fw] ^= 0x10

    def mem_read(self, ch, off, n):
        r = self.m[off:off + n].copy()
        self.reads += 1
        if self.fr is not None and self.reads == 1:
            r[self.fr] ^= 0x03
        return r


def run(**kw):
    t = Mem(1 << 16, **kw)
    offs = range(0, 1 << 16, 1 << 14)
    dd.wr_write(t, 0, offs, 1 << 14, 7)
    return dd.wr_read(t, 0, offs, 1 << 14, 7)


def test_wr_clean():
    r = run()
    assert (r["first"], r["second"], r["both"]) == (0, 0, 0) and dd.wr_verdict(r) == "clean"


def test_wr_stored_wrong():
    r = run(flip_write=8 * 5 + 2)              # byte lane 2 of word 5 of every piece, bit 4
    assert (r["first"], r["second"], r["both"]) == (4, 4, 4)
    assert dd.wr_verdict(r).startswith("stored wrong")
    assert r["lanes"].tolist() == [0, 0, 4, 0, 0, 0, 0, 0] and np.flatnonzero(r["bits"]).tolist() == [20]
    assert r["at"][0] == (40, 0x10 << 16)


def test_wr_read_wrong():
    r = run(flip_read=8 * 9 + 7)               # the first read only
    assert (r["first"], r["second"], r["both"]) == (1, 0, 0)
    assert dd.wr_verdict(r).startswith("read wrong")
    assert np.flatnonzero(r["bits"]).tolist() == [56, 57]


def test_lane_bits_and_line():
    bits = (1 << (8 * 8 + 3)) | (1 << (8 * 8 + 6)) | (1 << (8 * 7 + 0))
    assert dd.lane_bits(bits) == {7: [0], 8: [3, 6]}
    r = {"lanes": [0] * 7 + [2, 9], "bits": bits, "runs": 3}
    line = dd.lanes_line(1, r, 20)
    assert line.startswith("channel 1: 11 lane-beats wrong in 3 of 20 reads")
    assert "{7: 2, 8: 9}" in line and "{7: [0], 8: [3, 6]}" in line


def test_split_verdict():
    split_verdict = dd.split_verdict
    z = lambda: [0, 0, 0]
    sp = {"run": z(), "read 1": z(), "read 2": z(), "read 3": z()}
    assert split_verdict(sp) == "clean"
    sp["read 1"] = [2, 2, 0]
    assert "C2H" in split_verdict(sp)
    sp["read 2"] = [1, 1, 0]
    assert "stored wrong" in split_verdict(sp)
    sp = {"run": [1, 1, 0], "read 1": z(), "read 2": z(), "read 3": z()}
    assert "core" in split_verdict(sp)

