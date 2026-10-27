"""Legacy binary Qwen3 Bonsai 4B experiment (not Bonsai 2). Use bonsai2_validate.py for ternary."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=('checkpoint', 'reference', 'prebuild', 'board'),
                    default='checkpoint')
    ap.add_argument('--model', type=Path, default=ROOT / 'models/Bonsai-4B-mlx-1bit')
    ap.add_argument('--out', type=Path, default=ROOT / 'build/bonsai4b')
    ap.add_argument('--cap', type=int, default=512)
    ap.add_argument('--dev', default='/dev/xdma0')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from opentpu.llm import load_spec
    from opentpu.llm.qwen3 import load_weights
    spec, weights = load_spec(a.model), load_weights(a.model)
    config_hash = hashlib.sha256((a.model / 'config.json').read_bytes()).hexdigest()
    if (spec.layers, spec.hidden, spec.vocab) != (36, 2560, 151669):
        raise ValueError('Expected the pinned Qwen3 Bonsai 4B checkpoint')
    t0 = time.monotonic()

    if a.mode == 'checkpoint':
        w = weights['model.layers.0.self_attn.q_proj.weight']
        assert w.shape == (4096, 2560) and np.isfinite(w).all()
        print(f'Checkpoint decoder OK: {len(weights)} tensors, q_proj {w.shape}', flush=True)
        return 0

    if a.mode == 'reference':
        import torch
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        cfg = Qwen3Config(**json.loads((a.model / 'config.json').read_text()))
        with torch.device('meta'):
            model = Qwen3ForCausalLM(cfg)
        model.to_empty(device='cpu')
        model.tie_weights()
        model.eval()
        print('Building independent CPU float32 reference from decoded checkpoint', flush=True)
        with torch.no_grad():
            for n, p in model.named_parameters():
                value = weights[n]
                if tuple(value.shape) != tuple(p.shape):
                    raise ValueError(f'{n}: checkpoint {value.shape}, model {tuple(p.shape)}')
                p.copy_(torch.from_numpy(value))
                if n.endswith('input_layernorm.weight'):
                    print('Loaded', n, flush=True)
            tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
            ids = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': 'What is 2 plus 2? Answer with the number only.'}],
                add_generation_prompt=True, enable_thinking=False, tokenize=False)
            ids = tokenizer.encode(ids, add_special_tokens=False)
            ids = getattr(ids, 'ids', ids)
            logits = model(torch.tensor([ids])).logits[0, -1].float().numpy()
        np.savez(a.out / 'reference.npz', tokens=np.asarray(ids, np.int64), logits=logits,
                 config_sha256=np.asarray(config_hash))
        data = {'mode': 'CPU float32 reference', 'prompt_tokens': len(ids),
                'next_token': int(logits.argmax()), 'seconds': time.monotonic() - t0,
                'config_sha256': config_hash}
        (a.out / 'reference.json').write_text(json.dumps(data, indent=2)+'\n')
        print(json.dumps(data, indent=2), flush=True)
        return 0

    if a.mode == 'prebuild':
        from opentpu.isasim import board_config
        from opentpu.host.prebuild import prebuild
        from opentpu import qcache
        cfg = board_config(MCOLS=8, TMEM_WORDS=131072, TPAIR=True, TQUAD=True,
                           TFN=True, SEQ_WC=True, DRAM_BYTES=4 << 30)
        stop = threading.Event()
        def progress():
            while not stop.wait(30):
                print(f'Prebuild {time.monotonic()-t0:.0f}s; quantizer {dict(qcache.stats)}',
                      flush=True)
        threading.Thread(target=progress, daemon=True).start()
        print('CPU image preparation: fp4 projections, int8 head; no FPGA access', flush=True)
        try:
            data = prebuild(cfg, spec, weights, 'fp4', 'int8', a.cap)
        finally:
            stop.set()
        data.update(seconds=time.monotonic()-t0, config_sha256=config_hash,
                    device_format='fp4', head_format='int8', fpga_run=False)
        (a.out / 'prebuild.json').write_text(json.dumps(data, indent=2)+'\n')
        print(json.dumps(data, indent=2), flush=True)
        return 0

    # An independent test case: counters cleared only after the new image has been loaded;
    # the preexisting counts are retained in the report. Any new corrected/DED error fails.
    ref = np.load(a.out / 'reference.npz')
    if str(ref['config_sha256']) != config_hash:
        raise ValueError('Reference belongs to a different checkpoint config')
    ids, expected = ref['tokens'].tolist(), ref['logits'].astype(np.float64)
    if len(ids) >= a.cap or expected.shape != (spec.vocab,):
        raise ValueError('Reference dimensions/capacity do not match this run')
    from opentpu.host.chat import make_backend
    from opentpu.host import memcal
    from opentpu.llm.qwen3 import Engine
    be, cfg = make_backend('board', spec, a.cap, a.dev, 'Bonsai-4B-mlx-1bit',
                           lookup=True, wformat='fp4', head_format='int8')
    eng = None
    data = {'mode': 'FPGA board', 'device_format': 'fp4', 'head_format': 'int8',
            'config_sha256': config_hash, 'passed': False}
    try:
        eng = Engine(spec, weights, cap=a.cap, cfg=cfg, backend=be, resident=True,
                     pipeline=False, wformat='fp4', head_format='int8')
        transport = eng.backend.board.t
        data['prior_ecc'] = memcal.ecc(transport, clear=True)
        actual = eng.prefill(ids).astype(np.float64)
        if not np.isfinite(actual).all():
            raise RuntimeError('Nonfinite FPGA logits')
        den = np.linalg.norm(actual) * np.linalg.norm(expected)
        cosine = float(np.dot(actual, expected) / den) if den else 0.0
        data.update(cosine=cosine, fpga_next_token=int(actual.argmax()),
                    reference_next_token=int(expected.argmax()),
                    image_bytes=eng.image.nbytes, prompt_tokens=len(ids),
                    build_id=eng.backend.info.get('build_id'),
                    core_khz=eng.backend.info.get('core_khz'))
        data['ecc'] = memcal.ecc(transport)
        ecc_clean = data['ecc'] is not None and all(c == 0 and d == 0 for c, d in data['ecc'])
        data['passed'] = bool(ecc_clean and cosine >= 0.99)
    except Exception as e:
        data['error'] = f'{type(e).__name__}: {e}'
        if eng is not None:
            data['ecc'] = memcal.ecc(eng.backend.board.t)
    finally:
        if eng is not None:
            eng.backend.close()
    data['seconds'] = time.monotonic()-t0
    (a.out / 'board_check.json').write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps(data, indent=2), flush=True)
    return 0 if data['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
