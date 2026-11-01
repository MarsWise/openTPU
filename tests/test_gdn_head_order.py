import numpy as np
import pytest
from types import SimpleNamespace
from dataclasses import dataclass

from opentpu.llm.gdn_head_order import grouped_indices, reorder_gdn_tensor
from opentpu.llm.bonsai2 import hadamard
from opentpu.llm.gdn_head_order import GroupedGDNWeights


def test_grouped_indices():
    np.testing.assert_array_equal(grouped_indices(3, 6), [0, 3, 1, 4, 2, 5])


@pytest.mark.parametrize("name,head_width", [
    ("linear_attn.in_proj_z.weight", 2), ("linear_attn.in_proj_a.weight", 1),
    ("linear_attn.in_proj_b.weight", 1), ("linear_attn.A_log", 1),
    ("linear_attn.dt_bias", 1),
])
def test_value_head_rows_are_reordered(name, head_width):
    x = np.arange(6 * head_width * 3).reshape(6 * head_width, 3)
    expected = x.reshape(6, head_width, 3)[[0, 3, 1, 4, 2, 5]].reshape(x.shape)
    np.testing.assert_array_equal(reorder_gdn_tensor(name, x, 3, 6, 2, 2), expected)


@pytest.mark.parametrize("name", ["linear_attn.in_proj_qkv.weight", "linear_attn.conv1d.weight"])
def test_q_and_k_are_unchanged_but_v_is_reordered(name):
    x = np.arange((2 * 3 * 2 + 6 * 2) * 3).reshape(-1, 3)
    y = reorder_gdn_tensor(name, x, 3, 6, 2, 2)
    np.testing.assert_array_equal(y[:12], x[:12])
    np.testing.assert_array_equal(y[12:], x[12:].reshape(6, 2, 3)[[0, 3, 1, 4, 2, 5]].reshape(-1, 3))


def test_bad_geometry_and_unknown_tensor_rejected():
    with pytest.raises(ValueError):
        grouped_indices(3, 5)
    with pytest.raises(ValueError):
        reorder_gdn_tensor("linear_attn.in_proj_z.weight", np.zeros((5, 3)), 3, 6, 2, 2)
    with pytest.raises(ValueError):
        reorder_gdn_tensor("linear_attn.out_proj.weight", np.zeros((12, 3)), 3, 6, 2, 2)


def test_grouped_input_conversion_matches_tiled_recurrence_and_projection():
    rng = np.random.default_rng(23)
    nk, nv, dk, dv = 4, 8, 4, 4
    q, k = (rng.normal(size=(nk, dk)).astype(np.float32) for _ in range(2))
    v = rng.normal(size=(nv, dv)).astype(np.float32)
    beta = rng.random(nv, dtype=np.float32)
    z = rng.normal(size=(nv, dv)).astype(np.float32)
    state = rng.normal(size=(nv, dk, dv)).astype(np.float32)
    signs = rng.choice(np.float32([-1, 1]), nv * dv)
    weight = rng.normal(size=(5, nv * dv)).astype(np.float32)

    def run(order, heads):
        # A representative gated DeltaNet update; reorder recurrent state too.
        kk, qq = k[heads], q[heads]
        delta = (v[order] - np.einsum("hij,hi->hj", state[order], kk)) * beta[order, None]
        updated = state[order] + kk[:, :, None] * delta[:, None, :]
        return np.einsum("hij,hi->hj", updated, qq) * z[order]

    tiled = run(np.arange(nv), np.arange(nv) % nk)
    grouped = run(grouped_indices(nk, nv), np.arange(nv) // (nv // nk))
    np.testing.assert_allclose(grouped, tiled[grouped_indices(nk, nv)], rtol=1e-6, atol=1e-6)
    projected = hadamard(grouped.reshape(1, -1), 8, signs) @ weight.T
    baseline = hadamard(tiled.reshape(nv // nk, nk, dv).transpose(1, 0, 2).reshape(1, -1),
                        8, signs) @ weight.T
    np.testing.assert_array_equal(projected, baseline)


def test_experimental_wrapper_converts_inputs_only_and_preserves_source():
    class Weights(dict):
        g = SimpleNamespace(kv={"prism.hadamard.gdn_v_grouped": True})
        def signs_for(self, width):
            return np.ones(width, np.float32)
    @dataclass(frozen=True)
    class Spec:
        lin_nk: int = 3
        lin_heads: int = 6
        lin_dk: int = 2
        lin_dv: int = 2
        lin_tiled: bool = True
    z = np.arange(24).reshape(12, 2)
    out = np.arange(24).reshape(2, 12)
    source = Weights({"model.layers.0.linear_attn.in_proj_z.weight": z,
                      "model.layers.0.linear_attn.out_proj.weight": out})
    wrapped = GroupedGDNWeights(source, Spec())
    assert wrapped.spec.lin_tiled is False
    np.testing.assert_array_equal(wrapped[next(iter(source))], z.reshape(6, 2, 2)[[0, 3, 1, 4, 2, 5]].reshape(z.shape))
    np.testing.assert_array_equal(wrapped["model.layers.0.linear_attn.out_proj.weight"], out)
    np.testing.assert_array_equal(z, np.arange(24).reshape(12, 2))
    np.testing.assert_array_equal(wrapped.signs_for(2), [1, 1])


def test_experimental_wrapper_rejects_missing_contract():
    source = SimpleNamespace(g=SimpleNamespace(kv={}))
    spec = SimpleNamespace(lin_nk=3, lin_heads=6, lin_dk=2, lin_dv=2, lin_tiled=True)
    with pytest.raises(ValueError, match="gdn_v_grouped"):
        GroupedGDNWeights(source, spec)


def test_reordered_packed_ternary_preserves_group_scales():
    from opentpu.quant import pack_ternary_gsc
    rng = np.random.default_rng(7)
    nk, nv, dk, dv, width = 4, 8, 4, 4, 256
    rows = nv * dv
    codes = rng.integers(-1, 2, size=(rows, width), dtype=np.int8)
    codes[:, ::128] = 1  # every block contains a nonzero scale witness
    scales = rng.uniform(0.01, 1.0, size=(rows, 2)).astype(np.float32)
    values = (codes.reshape(rows, 2, 128) * scales[:, :, None]).reshape(rows, width)
    converted = reorder_gdn_tensor("linear_attn.in_proj_z.weight", values, nk, nv, dk, dv)
    packed, scale_bytes = pack_ternary_gsc(values)
    converted_packed, converted_scales = pack_ternary_gsc(converted)
    ix = (grouped_indices(nk, nv)[:, None] * dv + np.arange(dv)).reshape(-1)
    np.testing.assert_array_equal(converted_packed, packed[ix])
    np.testing.assert_array_equal(converted_scales, scale_bytes[ix])
