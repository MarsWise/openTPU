"""Independent bit vectors and real-container checks for the MLX affine checkpoint loader."""
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from opentpu.llm.mlx_weights import MLXWeights, unpack_u1
from opentpu.llm.qwen3 import Spec, load_weights, rope_tables
from opentpu.llm.yarn import parameters


def test_u1_low_bit_first_and_group_scales():
    w = np.array([[1, 0x80000000, 0x55555555, 0xAAAAAAAA, 0xFFFFFFFF, 0, 1, 2]], np.uint32)
    s = np.array([[2, 6]], np.float16)
    b = np.array([[-1, -3]], np.float16)
    got = unpack_u1(w, s, b)
    # Independent scalar extraction, including the first/last bit of each word and group.
    expected = [(2 if i < 128 else 6) * ((int(w[0, i // 32]) // 2 ** (i % 32)) % 2)
                + (-1 if i < 128 else -3) for i in range(256)]
    np.testing.assert_array_equal(got[0], expected)
    assert got.dtype == np.float32


@pytest.mark.parametrize('change', ['dtype', 'width', 'shape', 'nan', 'group'])
def test_u1_invalid_inputs_are_rejected(change):
    w, s, b = np.ones((2, 4), np.uint32), np.ones((2, 1)), np.zeros((2, 1))
    group = 128
    if change == 'dtype': w = w.astype(np.int32)
    if change == 'width': w = w[:, :3]
    if change == 'shape': s = s[:, :0]
    if change == 'nan': b[0, 0] = np.nan
    if change == 'group': group = 64
    with pytest.raises(ValueError):
        unpack_u1(w, s, b, group)


def checkpoint(path, *, omit=None, bits=1):
    (path / 'config.json').write_text(json.dumps(dict(model_type='qwen3',
        quantization=dict(group_size=128, bits=bits))))
    tensors = {'model.embed_tokens.weight': np.array([[0, 1, 2, 0xFFFFFFFF]], np.uint32),
               'model.embed_tokens.scales': np.array([[0.5]], np.float16),
               'model.embed_tokens.biases': np.array([[-0.25]], np.float16),
               'model.norm.weight': np.array([1.25, 0.75], np.float16)}
    if omit: tensors.pop(omit)
    save_file(tensors, str(path / 'model.safetensors'))


def test_lazy_reader_and_loader_dispatch(tmp_path):
    checkpoint(tmp_path)
    w = load_weights(tmp_path)
    assert isinstance(w, MLXWeights)
    assert set(w) == {'model.embed_tokens.weight', 'model.norm.weight'}
    np.testing.assert_array_equal(w['model.norm.weight'], [1.25, 0.75])
    a = w['model.embed_tokens.weight']
    assert a.shape == (1, 128) and a[0, 0] == -0.25 and a[0, 32] == 0.25
    assert w['model.embed_tokens.weight'] is a
    assert 'model.embed_tokens.scales' not in w


@pytest.mark.parametrize('omit', ['model.embed_tokens.scales', 'model.embed_tokens.biases'])
def test_missing_affine_tensor_is_refused(tmp_path, omit):
    checkpoint(tmp_path, omit=omit)
    with pytest.raises(ValueError, match='Missing MLX'):
        MLXWeights(tmp_path)


def test_unsupported_mlx_format_is_refused(tmp_path):
    checkpoint(tmp_path, bits=2)
    with pytest.raises(ValueError, match='1-bit'):
        load_weights(tmp_path)


def test_yarn_unit_factor_keeps_default_rope():
    s = Spec(hidden=128, layers=1, n_q=4, n_kv=2, head_dim=32, ffn=256, vocab=32)
    from dataclasses import replace
    y = replace(s, **parameters({'max_position_embeddings': 8192},
                {'factor': 1, 'original_max_position_embeddings': 8192}, 32, s.theta))
    for pos in (0, 1, 8191):
        for a, b in zip(rope_tables(s, pos), rope_tables(y, pos)):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('params', [dict(factor=0), dict(factor=float('nan')),
    dict(original_max_position_embeddings=0), dict(beta_fast=-1),
    dict(attention_factor=float('inf')), dict(partial_rotary_factor=0.5)])
def test_invalid_yarn_is_refused(params):
    p = dict(factor=4, original_max_position_embeddings=8192, **{})
    p.update(params)
    with pytest.raises(ValueError):
        parameters({'max_position_embeddings': 32768}, p, 128, 5000000)


def test_bonsai_yarn_matches_transformers_golden():
    """Golden from the host's installed Transformers _compute_yarn_parameters, not this code."""
    ref = json.loads((Path(__file__).parent / 'data/bonsai_yarn_reference.json').read_text(encoding='utf-8-sig'))
    kw = parameters({'max_position_embeddings': 32768},
                    {'factor': 4, 'original_max_position_embeddings': 8192}, 128, 5000000)
    base = 1 / (5000000.0 ** (np.arange(64) * 2 / 128))
    inv = base / kw['rope_div']
    np.testing.assert_allclose(inv, ref['inv_freq'], rtol=2e-7, atol=1e-10)
    assert kw['rope_scale'] == pytest.approx(ref['attention_scale'], abs=1e-12)


def test_packed_tiny_qwen3_forward_matches_dense_weights(tmp_path):
    """A whole decoder forward through the lazy checkpoint decoder, with YaRN active."""
    from opentpu.llm.qwen3 import reference_logits
    cfg = dict(model_type='qwen3', hidden_size=128, num_hidden_layers=1,
               num_attention_heads=4, num_key_value_heads=2, head_dim=32,
               intermediate_size=256, vocab_size=64, rms_norm_eps=1e-6,
               tie_word_embeddings=True, max_position_embeddings=32768,
               rope_theta=5000000, rope_scaling=dict(rope_type='yarn', factor=4,
               original_max_position_embeddings=8192),
               quantization=dict(group_size=128, bits=1))
    (tmp_path / 'config.json').write_text(json.dumps(cfg))
    rng = np.random.default_rng(7)
    dense, stored = {}, {}

    def matrix(name, n, k):
        bits = rng.integers(0, 2, (n, k), np.uint8)
        scale = np.full((n, k // 128), np.float16(0.02), np.float16)
        dense[name + '.weight'] = ((2 * bits.astype(np.float32) - 1) *
                                   np.repeat(scale.astype(np.float32), 128, axis=1))
        stored[name + '.weight'] = np.packbits(bits, axis=1, bitorder='little').view('<u4')
        stored[name + '.scales'] = scale * np.float16(2)
        stored[name + '.biases'] = -scale

    matrix('model.embed_tokens', 64, 128)
    p = 'model.layers.0.'
    for n, a, b in [('self_attn.q_proj',128,128), ('self_attn.k_proj',64,128),
                    ('self_attn.v_proj',64,128), ('self_attn.o_proj',128,128),
                    ('mlp.gate_proj',256,128), ('mlp.up_proj',256,128),
                    ('mlp.down_proj',128,256)]:
        matrix(p+n, a, b)
    for n, size in [('model.norm.weight',128), (p+'input_layernorm.weight',128),
                    (p+'post_attention_layernorm.weight',128),
                    (p+'self_attn.q_norm.weight',32), (p+'self_attn.k_norm.weight',32)]:
        dense[n] = stored[n] = np.ones(size, np.float32)
    save_file(stored, str(tmp_path / 'model.safetensors'))
    spec = Spec.from_hf(tmp_path)
    got = reference_logits(spec, load_weights(tmp_path), [1, 7, 3, 12])
    expected = reference_logits(spec, dense, [1, 7, 3, 12])
    np.testing.assert_array_equal(got, expected)
