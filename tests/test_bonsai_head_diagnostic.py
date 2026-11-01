import numpy as np
import pytest

from tools.bonsai2_gguf_check import _grouped_out_columns_to_tiled


@pytest.mark.parametrize("nk,rep,hd", [(2, 2, 3), (3, 3, 2), (16, 2, 128)])
def test_tiled_to_grouped_before_projection(nk, rep, hd):
    rng = np.random.default_rng(5)
    weight = rng.normal(size=(5, nk * rep * hd)).astype(np.float32)
    tiled = rng.normal(size=(3, nk * rep * hd)).astype(np.float32)
    grouped = tiled.reshape(3, rep, nk, hd).transpose(0, 2, 1, 3).reshape(tiled.shape)
    actual = tiled @ _grouped_out_columns_to_tiled(weight, nk, nk * rep).T
    np.testing.assert_allclose(actual, grouped @ weight.T, rtol=1e-5, atol=2e-5)


@pytest.mark.parametrize("shape,nk,nv", [((3, 7), 2, 4), ((3, 8), 3, 4), ((3, 8), 0, 4)])
def test_bad_geometry_is_rejected(shape, nk, nv):
    with pytest.raises(ValueError):
        _grouped_out_columns_to_tiled(np.zeros(shape), nk, nv)
