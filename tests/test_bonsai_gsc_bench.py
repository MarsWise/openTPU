import numpy as np
import pytest

from opentpu import Config
from tools.bench_bonsai_gsc import isa_result, make_case


@pytest.mark.parametrize('kb', [1, 3, 4, 5, 8, 9])
@pytest.mark.parametrize('mode', ['rsc', 'gsc'])
def test_uniform_scale_bench_case(kb, mode):
    cfg = Config(D=128, MCOLS=8, ACT_ROWS=16, ACT_BLOCKS=16,
                 TMEM_WORDS=131072, DRAM_BYTES=1 << 23, TGSC=True)
    image, _, _, checked, reference = make_case(cfg, 1, 7, kb, mode)
    actual = isa_result(cfg, image, checked, 1, 7)
    assert np.linalg.norm(actual - reference) / np.linalg.norm(reference) < .03


def test_modes_compare_identical_inputs_and_weights():
    cfg = Config(D=128, MCOLS=8, ACT_ROWS=16, ACT_BLOCKS=16,
                 TMEM_WORDS=131072, DRAM_BYTES=1 << 23, TGSC=True)
    results = []
    for mode in ('rsc', 'gsc'):
        image, _, _, checked, reference = make_case(cfg, 1, 7, 9, mode, seed=12)
        results.append((isa_result(cfg, image, checked, 1, 7), reference))
    np.testing.assert_array_equal(results[0][1], results[1][1])
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-5, atol=1e-5)


def test_bench_rejects_unsupported_memory_and_firmware():
    cfg = Config(D=128, MCOLS=8, ACT_ROWS=16, ACT_BLOCKS=16,
                 TMEM_WORDS=131072, DRAM_BYTES=1 << 23, TGSC=False)
    with pytest.raises(ValueError, match='TGSC'):
        make_case(cfg, 1, 7, 4, 'gsc')
