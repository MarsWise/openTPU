"""Bonsai 2's PTQ1_0 (PrismML's 1.75-bit ternary GGUF type 143) and PQ2_0 decoders, the GGUF
reader, and the lossless conversion to TMM T5 rows (opentpu/llm/ptq1.py). The element order
was checked against PQ2_0 on the real checkpoint (all 151 ternary tensors of
Ternary-Bonsai-2-2B, tools/bonsai_ptq1_check.py); here: round trips and the T5 math."""
import struct

import numpy as np
import pytest

from opentpu import gguf as GG
from opentpu import trits as T
from opentpu.llm import ptq1 as P


def test_round_trip_and_traversal():
    rng = np.random.default_rng(0)
    t = rng.integers(0, 3, (40, 128))
    d = rng.uniform(.01, .1, 40).astype(np.float16).astype(np.float32)
    raw = P.ptq1_encode(t, d)
    assert raw.size == 40 * 28
    t2, d2 = P.ptq1_trits(raw)
    np.testing.assert_array_equal(t2, t)
    np.testing.assert_array_equal(d2, d)
    # (qs is 24 bytes: the 32 stage is skipped) element 0..15: trit 0 of qs[0..15], 16..31
    # trit 1 of the same bytes .. 80..87 trit 0 of qs[16..23]; qh's four trits last
    b = raw.reshape(40, 28)
    np.testing.assert_array_equal(T.decode_bytes(b[:, :16])[..., 1], t[:, 16:32])
    np.testing.assert_array_equal(T.decode_bytes(b[:, 16:24])[..., 0], t[:, 80:88])
    np.testing.assert_array_equal(T.decode_bytes(b[:, 24:26])[..., 3], t[:, 126:128])


def test_pq2_codes():
    rng = np.random.default_rng(1)
    c = rng.integers(0, 3, (6, 128))
    d = np.float16(0.25)
    qs = np.zeros((6, 32), np.uint8)
    for j in range(128):
        qs[:, j >> 2] |= (c[:, j] << (2 * (j & 3))).astype(np.uint8)
    raw = np.concatenate([np.full((6, 2), 0, np.uint8) + np.frombuffer(d.tobytes(), np.uint8),
                          qs], 1).reshape(-1)
    c2, d2 = P.pq2_trits(raw)
    np.testing.assert_array_equal(c2, c)
    assert np.all(d2 == 0.25)


def test_to_t5_is_lossless():
    rng = np.random.default_rng(2)
    n, k = 8, 640
    t = rng.integers(0, 3, (n * k // 128, 128))
    d = rng.uniform(.01, .1, n * k // 128).astype(np.float16).astype(np.float32)
    rows, sc = P.to_t5(P.ptq1_encode(t, d), n, k)
    np.testing.assert_array_equal(T.unpack(rows, k), t.reshape(n, k) - 1)
    np.testing.assert_array_equal(sc, d.reshape(n, k // 128))
    assert rows.shape == (n, 128)                # five K-blocks in one D-byte chunk


def _gguf(path, tensors, kv):
    """A minimal GGUF v3 writer (the reader's test)."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, v in kv.items():
        head += s(k) + (struct.pack("<I", 8) + s(v) if isinstance(v, str)
                        else struct.pack("<II", 4, v))
    off, blobs = 0, []
    for name, (shape, typ, data) in tensors.items():
        head += s(name) + struct.pack("<I", len(shape)) + b"".join(
            struct.pack("<Q", x) for x in shape) + struct.pack("<IQ", typ, off)
        blobs.append(data.tobytes() + b"\0" * (-len(data.tobytes()) % 32))
        off += len(blobs[-1])
    head += b"\0" * (-len(head) % 32)
    path.write_bytes(head + b"".join(blobs))


def test_reader(tmp_path):
    rng = np.random.default_rng(3)
    t = rng.integers(0, 3, (4, 128))
    raw = P.ptq1_encode(t, np.ones(4, np.float32))
    norm = rng.normal(size=256).astype(np.float32)
    _gguf(tmp_path / "m.gguf", {"w": ((256, 2), GG.PTQ1_0, raw), "n": ((256,), GG.F32, norm)},
          {"general.architecture": "qwen35", "qwen35.block_count": 2})
    g = GG.GGUF(tmp_path / "m.gguf")
    assert g.kv["general.architecture"] == "qwen35" and g.kv["qwen35.block_count"] == 2
    np.testing.assert_array_equal(np.asarray(g.raw("w")), raw)
    np.testing.assert_array_equal(np.frombuffer(bytes(g.raw("n")), np.float32), norm)
    with pytest.raises(ValueError, match="GGUF"):
        (tmp_path / "x").write_bytes(b"NOPE" + bytes(20))
        GG.GGUF(tmp_path / "x")
