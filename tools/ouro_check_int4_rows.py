"""Offline numerical spot-check of converted Ouro weights; not model inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opentpu.quant import dequantize_mxu, quantize_mxu
from tools.ouro_convert_int4 import verify_conversion


def check_rows(model: Path, output: Path, names: list[str], rows: int):
    if rows < 1:
        raise ValueError('rows must be positive')
    manifest = verify_conversion(model, output)
    result = []
    with safe_open(str(model / 'model.safetensors'), framework='pt', device='cpu') as source, \
            (output / 'weights.bin').open('rb') as weights, (output / 'scales.bin').open('rb') as scales:
        for name in names:
            rec = manifest['tensors'][name]
            if rec['format'] != 'int4':
                raise ValueError(f'{name}: not a matrix')
            n, k = rec['shape']
            n = min(rows, n)
            weights.seek(rec['weights_offset'])
            scales.seek(rec['scales_offset'])
            q = np.frombuffer(weights.read(n * rec['weights_row_bytes']), np.uint8).reshape(n, -1)
            s = np.frombuffer(scales.read(n * rec['scales_row_bytes']), np.uint8).reshape(n, -1)
            original = source.get_slice(name)[:n].float().numpy()
            want_q, want_s = quantize_mxu(original, 'int4', D=128)
            if not np.array_equal(q, want_q) or not np.array_equal(s, want_s):
                raise ValueError(f'{name}: converted bytes differ from MXU quantizer')
            decoded = dequantize_mxu(q, s, 'int4', k)
            diff = decoded - original
            denom = float(np.linalg.norm(original))
            result.append({'tensor': name, 'rows': n, 'cols': k,
                           'relative_l2': float(np.linalg.norm(diff) / denom) if denom else None,
                           'max_abs_error': float(np.max(np.abs(diff))),
                           'packed_bytes_match': True})
    return {'source_sha256': manifest['source_sha256'], 'format': manifest['format'],
            'runtime_supported': False, 'tensors': result}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path, help='existing converted image')
    ap.add_argument('--tensor', action='append', required=True)
    ap.add_argument('--rows', type=int, default=64)
    ap.add_argument('--report', type=Path)
    args = ap.parse_args()
    result = json.dumps(check_rows(args.model, args.out, args.tensor, args.rows), indent=2)
    if args.report:
        with args.report.open('x', encoding='utf-8') as f:
            f.write(result + '\n')
    print(result)


if __name__ == '__main__':
    main()
