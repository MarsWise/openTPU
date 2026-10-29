"""Wide elementwise tiles must account for the temporary binary result."""
import numpy as np
import pytest

from opentpu import Config
from opentpu.kernels import eltwise
from opentpu.runtime import Input, Output, compile_kernel, run


@pytest.mark.parametrize('rows,width,broadcast', [(17, 3072, False),
                                                  (17, 3072, True),
                                                  (1, 20480, False)])
def test_wide_binary_fits_tmem_and_matches_float(rows, width, broadcast):
    rng = np.random.default_rng(23)
    a = rng.uniform(-1, 1, (rows, width)).astype(np.float32)
    b = rng.uniform(-1, 1, width if broadcast else (rows, width)).astype(np.float32)
    cfg = Config(D=128, MCOLS=8, ACT_ROWS=16, ACT_BLOCKS=128,
                 TMEM_WORDS=131072, DRAM_BYTES=1 << 26)
    compiled, images = compile_kernel(eltwise.binary, cfg, a=Input(a), b=Input(b),
                                      y=Output((rows, width)), M=rows, N=width,
                                      op='mul', brow=broadcast)
    actual = run(compiled, images, 'isa').outputs['y']
    np.testing.assert_allclose(actual, a * b, rtol=1e-6, atol=1e-7)
