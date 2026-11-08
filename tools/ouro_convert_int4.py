"""Convert an Ouro safetensors checkpoint to openTPU MXU INT4 rows.

Offline conversion only. The resulting files are not yet loadable by the
inference runtime; Ouro's recurrent executor and per-step KV cache are pending.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opentpu.llm.ouro import OuroSpec
from opentpu.quant import quantize_mxu, row_bytes, scale_bytes


ALIGNMENT = 128  # Keep MXU rows aligned; runtime placement still needs its own validation.


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda: src.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def align_file(stream):
    padding = (-stream.tell()) % ALIGNMENT
    if padding:
        stream.write(bytes(padding))


def verify_conversion(model: Path, output: Path):
    """Check an offline v2 image before use; this does not establish runtime support."""
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    spec = OuroSpec.from_hf(model)
    if (manifest.get('format') != 'opentpu-ouro-int4-preparation-v2' or
            manifest.get('runtime_supported') is not False or manifest.get('D') != 128 or
            manifest.get('alignment_bytes') != ALIGNMENT or manifest.get('steps') != spec.steps):
        raise ValueError('unsupported Ouro preparation manifest')
    if (manifest.get('source_sha256') != sha256_file(model / 'model.safetensors') or
            manifest.get('config_sha256') != sha256_file(model / 'config.json')):
        raise ValueError('Ouro checkpoint/config checksum mismatch')
    files = {}
    for name in ('weights.bin', 'scales.bin'):
        path = output / name
        actual = {'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
        if manifest.get('files', {}).get(name) != actual:
            raise ValueError(f'{name}: checksum or size mismatch')
        files[name] = actual['bytes']
    expected = required_shapes(spec)
    records = manifest.get('tensors', {})
    if set(records) != set(expected):
        raise ValueError('Ouro tensor names mismatch')
    intervals = {name: [] for name in files}
    for name, shape in expected.items():
        rec = records[name]
        if tuple(rec.get('shape', ())) != shape or rec.get('format') != ('int4' if len(shape) == 2 else 'fp32'):
            raise ValueError(f'{name}: shape or format mismatch')
        if len(shape) == 2:
            rs, srs = row_bytes(shape[1], 'int4', 128), scale_bytes(shape[1], 'int4', 128)
            if rec.get('weights_row_bytes') != rs or rec.get('scales_row_bytes') != srs:
                raise ValueError(f'{name}: row stride mismatch')
            spans = [('weights.bin', 'weights', shape[0] * rs),
                     ('scales.bin', 'scales', shape[0] * srs)]
        else:
            spans = [('weights.bin', 'weights', shape[0] * 4)]
        for filename, prefix, size in spans:
            off = rec.get(prefix + '_offset')
            if (not isinstance(off, int) or off < 0 or off % ALIGNMENT or
                    rec.get(prefix + '_bytes') != size or off + size > files[filename]):
                raise ValueError(f'{name}: invalid {prefix} span')
            intervals[filename].append((off, off + size, name))
    for filename, spans in intervals.items():
        spans.sort()
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            raise ValueError(f'{filename}: overlapping tensor spans')
    return manifest


def required_shapes(spec):
    h, f, d, v = spec.hidden, spec.ffn, spec.kv_heads * spec.head_dim, spec.vocab
    shapes = {'model.embed_tokens.weight': (v, h), 'lm_head.weight': (v, h),
              'model.norm.weight': (h,), 'model.early_exit_gate.weight': (1, h),
              'model.early_exit_gate.bias': (1,)}
    for layer in range(spec.layers):
        p = f'model.layers.{layer}.'
        for name, shape in {'self_attn.q_proj.weight': (h, h), 'self_attn.k_proj.weight': (d, h),
                            'self_attn.v_proj.weight': (d, h), 'self_attn.o_proj.weight': (h, h),
                            'mlp.gate_proj.weight': (f, h), 'mlp.up_proj.weight': (f, h),
                            'mlp.down_proj.weight': (h, f), 'input_layernorm.weight': (h,),
                            'input_layernorm_2.weight': (h,), 'post_attention_layernorm.weight': (h,),
                            'post_attention_layernorm_2.weight': (h,)}.items():
            shapes[p + name] = shape
    return shapes


def convert(model: Path, output: Path, chunk_rows: int = 256):
    from safetensors import safe_open
    if chunk_rows <= 0 or chunk_rows > 4096:
        raise ValueError('chunk_rows must be 1..4096')
    spec = OuroSpec.from_hf(model)
    source = model / 'model.safetensors'
    if not source.is_file():
        raise FileNotFoundError(source)
    expected = required_shapes(spec)
    # Validate the whole checkpoint before creating output or converting anything.
    with safe_open(str(source), framework='pt', device='cpu') as f:
        names = set(f.keys())
        if names != set(expected):
            raise ValueError(f'Ouro tensor mismatch: missing={sorted(set(expected)-names)}, '
                             f'extra={sorted(names-set(expected))}')
        for name, shape in expected.items():
            if tuple(f.get_slice(name).get_shape()) != shape:
                raise ValueError(f'{name}: expected shape {shape}')
            if len(shape) == 2 and shape[1] % 128:
                raise ValueError(f'{name}: INT4 K must be divisible by 128')
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    output.mkdir(parents=True)
    manifest = {'format': 'opentpu-ouro-int4-preparation-v2', 'runtime_supported': False,
                'source': str(source.resolve()), 'source_sha256': sha256_file(source),
                'config_sha256': sha256_file(model / 'config.json'),
                'D': 128, 'alignment_bytes': ALIGNMENT,
                'steps': spec.steps, 'tensors': {}}
    with safe_open(str(source), framework='pt', device='cpu') as f, \
            (output / 'weights.bin').open('wb') as wout, (output / 'scales.bin').open('wb') as sout:
        for name in sorted(expected):
            if not re.fullmatch(r'[A-Za-z0-9_.]+', name):
                raise ValueError(f'unsafe tensor name {name!r}')
            shape = expected[name]
            t = f.get_slice(name)
            record = {'shape': shape}
            align_file(wout)
            if len(shape) == 2:
                align_file(sout)
                rs = row_bytes(shape[1], 'int4', D=128)
                srs = scale_bytes(shape[1], 'int4', D=128)
                record.update(format='int4', weights_offset=wout.tell(), scales_offset=sout.tell(),
                              weights_row_bytes=rs, scales_row_bytes=srs)
                for at in range(0, shape[0], chunk_rows):
                    a = t[at:min(at + chunk_rows, shape[0])].float().numpy()
                    if not np.isfinite(a).all():
                        raise ValueError(f'{name}: non-finite weights')
                    q, s = quantize_mxu(a, 'int4', D=128)
                    if q.shape != (len(a), rs) or s.shape != (len(a), srs):
                        raise ValueError(f'{name}: unexpected MXU row stride')
                    wout.write(q.tobytes())
                    sout.write(s.tobytes())
                record['weights_bytes'] = wout.tell() - record['weights_offset']
                record['scales_bytes'] = sout.tell() - record['scales_offset']
            else:
                a = t[:].float().numpy().astype('<f4', copy=False)
                if not np.isfinite(a).all():
                    raise ValueError(f'{name}: non-finite vector')
                record.update(format='fp32', weights_offset=wout.tell(), weights_bytes=a.nbytes)
                wout.write(a.tobytes())
            manifest['tensors'][name] = record
            print(name, record['format'], flush=True)
    manifest['files'] = {name: {'bytes': (output / name).stat().st_size,
                                 'sha256': sha256_file(output / name)}
                         for name in ('weights.bin', 'scales.bin')}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--chunk-rows', type=int, default=256)
    ap.add_argument('--verify-only', action='store_true', help='verify an existing v2 conversion')
    args = ap.parse_args()
    if args.verify_only:
        manifest = verify_conversion(args.model, args.out)
        print(f"Verified {len(manifest['tensors'])} tensors; runtime support remains unavailable.")
    else:
        convert(args.model, args.out, args.chunk_rows)


if __name__ == '__main__':
    main()
