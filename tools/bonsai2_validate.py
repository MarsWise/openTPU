"""Native ternary Bonsai 2 4B validation. Only --mode board opens the FPGA."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def configuration():
    from opentpu.isasim import board_config
    return board_config(MCOLS=8, TMEM_WORDS=131072, TPAIR=True, TQUAD=True, TFN=True,
                        TGSC=True, PAIR=True, DSTEP=True, STREAM=True, SEQ_WC=True,
                        DRAM_BYTES=4 << 30)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=('checkpoint', 'compile', 'prebuild', 'reference', 'isa', 'board'),
                    default='checkpoint')
    ap.add_argument('--model', type=Path, default=ROOT/'models/Ternary-Bonsai-2-4B-mlx-2bit')
    ap.add_argument('--out', type=Path, default=ROOT/'build/bonsai2-4b')
    ap.add_argument('--cap', type=int, default=512)
    ap.add_argument('--dev', default='/dev/xdma0')
    ap.add_argument('--full-prompt', action='store_true',
                    help='ISA mode: process the complete reference prompt, including recurrent state')
    args = ap.parse_args()
    if args.full_prompt and args.mode != 'isa':
        ap.error('--full-prompt is only valid with --mode isa')
    args.out.mkdir(parents=True, exist_ok=True)
    from opentpu.llm import load_spec
    from opentpu.llm.qwen3 import load_weights, ATTN_BLOCK
    from opentpu.llm.bonsai2 import MODEL_ID, REVISION, NativeWeights, NativeMatrix, hadamard
    from opentpu.target import arch_hash
    spec = load_spec(args.model)
    if (spec.layers, spec.hidden, spec.ffn, spec.vocab) != (32, 2560, 9216, 248320):
        raise ValueError('Expected ternary Bonsai 2 4B, Qwen3.5, 32 layers')
    cfg = configuration()
    image = spec.image(cfg, args.cap, rows=1, lookup=True)
    data = dict(model=MODEL_ID, expected_revision=REVISION, fpga_run=False,
                device_format='ternary_gsc', head_format='ternary_gsc',
                image_bytes=image.nbytes, free_bytes=cfg.DRAM_BYTES-image.nbytes,
                cfg_hash=f'{arch_hash(cfg):08x}',
                config_sha256=hashlib.sha256((args.model/'config.json').read_bytes()).hexdigest())
    t0 = time.monotonic()
    if args.mode == 'compile':
        programs = {'position0': image.compile_step(0, tok=4),
                    'resident_bucket1': image.compile_decode(1, 3)[0],
                    'greedy_generate': image.compile_generate(1, 3),
                    'resident_last_bucket': image.compile_decode((args.cap-1)//ATTN_BLOCK+1,
                                                                 args.cap-1)[0]}
        data['instructions'] = {k: list(map(len, ps)) for k, ps in programs.items()}
        limit = cfg.IMEM_WORDS//8
        if any(len(p) > limit for ps in programs.values() for p in ps):
            raise ValueError(f'Program exceeds IMEM ({limit} instructions): {data["instructions"]}')
    elif args.mode == 'checkpoint':
        weights = NativeWeights(args.model)
        checks = []
        for key in ('model.layers.0.linear_attn.in_proj_qkv.weight',
                    'model.layers.3.self_attn.q_proj.weight', 'model.embed_tokens.weight'):
            mat = weights[key]
            packed, scales = mat.tmm()
            checks.append(dict(name=key, shape=mat.shape, packed_bytes=packed.nbytes,
                               scales_bytes=scales.nbytes))
        data['matrices'] = checks
        data['tensors'] = len(weights)
    elif args.mode == 'reference':
        import torch
        from transformers import AutoTokenizer, Qwen3_5TextConfig, Qwen3_5ForCausalLM
        torch.set_num_threads(4)
        top = json.loads((args.model/'config.json').read_text())
        tc = Qwen3_5TextConfig(**top['text_config'])
        with torch.device('meta'):
            model = Qwen3_5ForCausalLM(tc)
        model.to_empty(device='cpu')
        model.tie_weights()
        native = NativeWeights(args.model)
        with torch.no_grad():
            for name, param in model.named_parameters():
                value = native[name]
                if isinstance(value, NativeMatrix):
                    trits, scales = value.unpack()
                    value_ = (trits.reshape(len(trits), -1, 128) * scales[..., None]).reshape(trits.shape)
                    value = hadamard(value_, value.block, value.signs, inverse=True)
                if param.shape != value.shape:
                    raise ValueError(f'Reference shape mismatch {name}: {param.shape}, {value.shape}')
                param.copy_(torch.from_numpy(value))
                if name.endswith('input_layernorm.weight'):
                    print('Reference loaded', name, flush=True)
            model.eval()
            tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
            text = tok.apply_chat_template([{'role':'user', 'content':'What is 2 plus 2? Answer with the number only.'}],
                                           add_generation_prompt=True, enable_thinking=False, tokenize=False)
            ids = tok.encode(text, add_special_tokens=False)
            ids = getattr(ids, 'ids', ids)
            all_logits = model(torch.tensor([ids], dtype=torch.long)).logits[0].float().numpy()
            logits = all_logits[-1]
        np.savez(args.out/'reference.npz', tokens=np.asarray(ids, np.int64), logits=logits,
                 first_logits=all_logits[0],
                 config_sha256=np.asarray(data['config_sha256']))
        data.update(prompt_tokens=len(ids), next_token=int(logits.argmax()),
                    reference='FP32 HF math with weights unfolded only for independent reference')
    elif args.mode == 'prebuild':
        from opentpu.host.prebuild import prebuild
        data.update(prebuild(cfg, spec, load_weights(args.model), 'ternary_gsc', 'ternary_gsc', args.cap))
    elif args.mode == 'isa':
        from opentpu.llm.qwen3 import Engine
        ref = np.load(args.out/'reference.npz')
        if str(ref['config_sha256']) != data['config_sha256']:
            raise ValueError('CPU reference belongs to another checkpoint')
        ids = ref['tokens'].tolist() if args.full_prompt else [int(ref['tokens'][0])]
        expected = ref['logits' if args.full_prompt else 'first_logits'].astype(np.float64)
        eng = Engine(spec, load_weights(args.model), cfg=cfg, cap=args.cap, backend='isa',
                     resident=True, pipeline=False, rows=1, wformat='ternary_gsc')
        for pos, token in enumerate(ids):
            actual = eng.step(token).astype(np.float64)
            print(f'ISA prompt token {pos+1}/{len(ids)} complete', flush=True)
        cosine = float(actual @ expected/(np.linalg.norm(actual)*np.linalg.norm(expected)))
        data.update(cosine=cosine, passed=bool(np.isfinite(actual).all() and cosine >= .99),
                    prompt_tokens=len(ids), first_token=ids[0], next_token=int(actual.argmax()),
                    reference_next_token=int(expected.argmax()))
        np.savez(args.out/('isa_prompt_logits.npz' if args.full_prompt else 'isa_logits.npz'),
                 actual=actual, expected=expected)
    else:
        from opentpu.host.chat import make_backend
        from opentpu.host import memcal
        from opentpu.host.ecc_guard import require_clean_ecc
        from opentpu.llm.qwen3 import Engine
        ref = np.load(args.out/'reference.npz')
        if str(ref['config_sha256']) != data['config_sha256']:
            raise ValueError('CPU reference belongs to another checkpoint')
        ids, expected = ref['tokens'].tolist(), ref['logits'].astype(np.float64)
        if len(ids) >= args.cap or expected.shape != (spec.vocab,):
            raise ValueError('CPU reference dimensions/capacity do not match')
        be, board_cfg = make_backend('board', spec, args.cap, args.dev, args.model.name,
                                     lookup=True, wformat='ternary_gsc', head_format='ternary_gsc',
                                     guard_ecc=True)
        if not board_cfg.TGSC:
            raise ValueError('New TGSC firmware is required before loading native Bonsai 2')
        eng = None
        data.update(fpga_run=True, passed=False)
        try:
            eng = Engine(spec, load_weights(args.model), cfg=board_cfg, backend=be,
                         cap=args.cap, resident=True, pipeline=False, rows=1,
                         wformat='ternary_gsc', head_format='ternary_gsc', prompt_runs=False)
            transport = eng.backend.board.t
            info = eng.backend.board.info()
            data.update(build_id=info.get('build_id'), core_khz=info.get('core_khz'),
                        actual_cfg_hash=(info.get('cfg') or {}).get('hash'),
                        pre_scrub_ecc=eng.backend.pre_scrub_ecc)
            data['prior_ecc'] = memcal.ecc(transport)
            data['prior_status'] = eng.backend.board.info()['status']
            require_clean_ecc(data['prior_ecc'], data['prior_status'])
            actual = eng.prefill(ids).astype(np.float64)
            data['ecc'] = memcal.ecc(transport)
            data['status'] = eng.backend.board.info()['status']
            require_clean_ecc(data['ecc'], data['status'])
            if not np.isfinite(actual).all():
                raise ValueError('Nonfinite FPGA logits')
            np.save(args.out/'board_logits.npy', actual)
            den = np.linalg.norm(actual)*np.linalg.norm(expected)
            cosine = float(actual @ expected/den) if den else 0.
            data.update(cosine=cosine, passed=bool(cosine >= .99),
                        next_token=int(actual.argmax()), reference_next_token=int(expected.argmax()))
        except Exception as e:
            data['error'] = f'{type(e).__name__}: {e}'
        finally:
            if eng is not None:
                eng.backend.close()
    data['seconds'] = round(time.monotonic()-t0, 3)
    name = 'isa_prompt' if args.full_prompt else args.mode
    (args.out/(name+'.json')).write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps(data, indent=2), flush=True)
    return 0 if data.get('passed', True) else 1


if __name__ == '__main__':
    raise SystemExit(main())
