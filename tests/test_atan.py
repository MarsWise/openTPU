"""The TFN atan (VOP 0x2A, Config.TATAN: table 8, atan on [0, 1]) and atan2 on the device
(kernels/atan2.py): accuracy against float64, ISA against its bit-exact reference, RTL == ISA."""
import math

import numpy as np
import pytest

from opentpu import Config
from opentpu import isa as I
from opentpu import language as ol
from opentpu import tfn as T
from opentpu.kernels import atan2 as A
from opentpu.runtime import Input, Output, launch


def test_the_first_tables_are_unchanged():
    """Table 8 is appended: tables 0..7 keep their ROM words (every other TFN's results)."""
    coef, tails = T.tables()
    assert coef.shape == (9, 256, 3) and tails.shape == (9, 2)
    fit, _ = T.fit_tables()
    np.testing.assert_array_equal(coef[8], fit[8])
    assert I.V_ATAN == 0x2A and T.func_code("atan") == I.V_ATAN


def test_atan_table_accuracy():
    x = np.linspace(0, 1, 200001, dtype=np.float32)
    y = T.tfn(T.BY_NAME["atan"], x).astype(np.float64)
    assert np.abs(y - np.arctan(x.astype(np.float64))).max() < 2e-7
    assert T.tfn(T.BY_NAME["atan"], np.float32([1.0, 5.0]))[0] == np.float32(math.pi / 4)


def test_atan2_reference_accuracy():
    rng = np.random.default_rng(3)
    y = rng.normal(size=20000).astype(np.float32) * 10 ** rng.uniform(-3, 3, 20000).astype(np.float32)
    x = rng.normal(size=20000).astype(np.float32) * 10 ** rng.uniform(-3, 3, 20000).astype(np.float32)
    y[:6] = [0, 1, -1, 0, 1, -1]
    x[:6] = [0, 0, 0, -1, 1, -1]
    got = A.reference(y, x).astype(np.float64)
    want = np.arctan2(y.astype(np.float64), x.astype(np.float64))
    assert np.abs(got - want).max() < 1e-6


@ol.jit
def _k(a, b, out, N: int):
    ol.store(out[0:N], A.atan2(ol.load(a[0:N]), ol.load(b[0:N])))


def _cfg():
    return Config(S=1, D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16, DRAM_BYTES=1 << 24, TFN=True,
                  DSTEP=True, VCMP=True, TATAN=True)


@pytest.mark.parametrize("backend", ["isa", "rtl"])
def test_atan2_kernel(backend, request):
    if backend == "rtl":
        request.getfixturevalue("have_verilator")
    rng = np.random.default_rng(5)
    N = 200
    y, x = rng.normal(size=(2, N)).astype(np.float32)
    y[:4], x[:4] = [0, 1, -1, 2], [0, 0, -3, -2]
    got = launch(_k, _cfg(), backend=backend, a=Input(y), b=Input(x), out=Output((N,)),
                 N=N).outputs["out"]
    np.testing.assert_array_equal(got.view(np.uint32), A.reference(y, x).view(np.uint32))
