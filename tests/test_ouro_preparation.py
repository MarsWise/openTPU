"""Ouro geometry, step/layer cache addressing and offline INT4 conversion."""
import hashlib
import json

import numpy as np
import pytest

from opentpu.llm.ouro import OuroSpec
from opentpu.quant import dequantize_mxu
from tools.ouro_convert_int4 import convert, required_shapes, verify_conversion


def config(layers=2):
    return dict(model_type='ouro', hidden_act='silu', num_hidden_layers=layers,
                layer_types=['full_attention'] * layers, hidden_size=128,
                intermediate_size=256, num_attention_heads=2, num_key_value_heads=2,
                head_dim=64, vocab_size=128, max_position_embeddings=512,
                rope_theta=1000000.0, rms_norm_eps=1e-6, total_ut_steps=4,
                tie_word_embeddings=False, rope_scaling=None)


def test_recurrent_kv_cache_indices_and_payload(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps(config()), encoding='utf-8')
    spec = OuroSpec.from_hf(tmp_path)
    assert [spec.cache_index(s, l) for s in range(4) for l in range(2)] == list(range(8))
    assert spec.kv_payload_bytes(512) == 4 * 2 * 512 * 2 * 2 * 64
    assert OuroSpec.from_hf(tmp_path, 1).kv_payload_bytes(512) * 4 == spec.kv_payload_bytes(512)
    with pytest.raises(ValueError):
        spec.cache_index(4, 0)
    with pytest.raises(ValueError):
        OuroSpec.from_hf(tmp_path, 5)


def test_native_int4_checkpoint_conversion_and_reconstruction(tmp_path):
    torch = pytest.importorskip('torch')
    st = pytest.importorskip('safetensors.torch')
    model = tmp_path / 'model'; model.mkdir()
    (model / 'config.json').write_text(json.dumps(config(1)), encoding='utf-8')
    spec = OuroSpec.from_hf(model)
    shapes = required_shapes(spec)
    rng = np.random.default_rng(71)
    tensors = {k: torch.from_numpy(rng.normal(0, .1, size=v).astype(np.float32))
               for k, v in shapes.items()}
    st.save_file(tensors, str(model / 'model.safetensors'))
    out = tmp_path / 'int4'
    manifest = convert(model, out, chunk_rows=13)
    assert manifest['runtime_supported'] is False
    assert len(manifest['source_sha256']) == 64
    assert manifest['format'] == 'opentpu-ouro-int4-preparation-v2'
    assert manifest['alignment_bytes'] == 128
    weights, scales = (out / 'weights.bin').read_bytes(), (out / 'scales.bin').read_bytes()
    for name, data in [('weights.bin', weights), ('scales.bin', scales)]:
        assert manifest['files'][name] == {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    for tensor_name, record in manifest['tensors'].items():
        assert record['weights_offset'] % 128 == 0
        if record['format'] == 'int4':
            assert record['scales_offset'] % 128 == 0
            assert record['weights_bytes'] == record['shape'][0] * record['weights_row_bytes']
            assert record['scales_bytes'] == record['shape'][0] * record['scales_row_bytes']
        else:
            data = weights[record['weights_offset']:record['weights_offset'] + record['weights_bytes']]
            np.testing.assert_array_equal(np.frombuffer(data, '<f4'), tensors[tensor_name].numpy())
    name = 'model.layers.0.self_attn.q_proj.weight'
    rec = manifest['tensors'][name]
    n, k = rec['shape']
    q = np.frombuffer(weights, np.uint8, rec['weights_bytes'], rec['weights_offset']).reshape(n, -1)
    sc = np.frombuffer(scales, np.uint8, rec['scales_bytes'], rec['scales_offset']).reshape(n, -1)
    decoded = dequantize_mxu(q, sc, 'int4', k)
    orig = tensors[name].numpy()
    assert np.linalg.norm(decoded - orig) / np.linalg.norm(orig) < .2
    assert verify_conversion(model, out)['source_sha256'] == manifest['source_sha256']
    from tools.ouro_check_int4_rows import check_rows
    sample = check_rows(model, out, [name], 7)
    assert sample['tensors'][0]['packed_bytes_match']
    assert sample['tensors'][0]['relative_l2'] < .2
    manifest_file = out / 'manifest.json'
    original_manifest = manifest_file.read_text(encoding='utf-8')
    tampered = json.loads(original_manifest)
    tampered['tensors']['model.early_exit_gate.bias']['weights_offset'] = 0
    manifest_file.write_text(json.dumps(tampered), encoding='utf-8')
    with pytest.raises(ValueError, match='overlapping'):
        verify_conversion(model, out)
    manifest_file.write_text(original_manifest, encoding='utf-8')
    # Chunking must not alter the final representation, including padding.
    out2 = tmp_path / 'int4-other-chunk-size'
    convert(model, out2, chunk_rows=29)
    assert weights == (out2 / 'weights.bin').read_bytes()
    assert scales == (out2 / 'scales.bin').read_bytes()
    damaged = bytearray((out2 / 'weights.bin').read_bytes())
    damaged[0] ^= 1
    (out2 / 'weights.bin').write_bytes(damaged)
    with pytest.raises(ValueError, match='checksum'):
        verify_conversion(model, out2)
    with pytest.raises(FileExistsError):
        convert(model, out)


def test_unsupported_config_is_rejected(tmp_path):
    c = config(); c['layer_types'][1] = 'sliding_attention'
    (tmp_path / 'config.json').write_text(json.dumps(c), encoding='utf-8')
    with pytest.raises(ValueError, match='full-attention'):
        OuroSpec.from_hf(tmp_path)
