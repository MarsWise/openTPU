#!/usr/bin/env python3
"""Run the SmolVLA FP32 reference on a CUDA machine, the openTPU run on the FPGA host.

Place beside tools/smolvla_run.py in the openTPU checkout on BOTH hosts.
This is a hybrid runner: board mode still executes numerical operations on the
host. It never programs an FPGA bitstream. ECC and board locking must be handled
by the caller; the sampled weight fingerprint is not a checkpoint hash.
"""
import argparse
import contextlib
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import smolvla_run as sm


def _leaves(items):
    for item in items:
        if isinstance(item, (tuple, list)):
            yield from _leaves(item)
        else:
            yield item


def input_fingerprint(x):
    h = hashlib.sha256()
    for t in _leaves(x):
        arr = t.detach().cpu().contiguous().numpy()
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def weight_fingerprint(model):
    """Small deterministic model fingerprint, NOT a full checkpoint hash."""
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        h.update(name.encode())
        h.update(str(tuple(param.shape)).encode())
        h.update(str(param.dtype).encode())
        flat = param.detach().reshape(-1)
        n = min(16, flat.numel())
        if n:
            h.update(flat[:n].float().cpu().numpy().astype('<f4').tobytes())
            h.update(flat[-n:].float().cpu().numpy().astype('<f4').tobytes())
    return h.hexdigest()


def to_device(obj, device):
    if isinstance(obj, list):
        return [to_device(i, device) for i in obj]
    if isinstance(obj, tuple):
        return tuple(to_device(i, device) for i in obj)
    return obj.to(device) if hasattr(obj, 'to') else obj


def take_scalar(z, key):
    return z[key].item() if key in z.files else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', required=True, choices=('reference', 'board'))
    ap.add_argument('--policy', default='lerobot/smolvla_base')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--ref', required=True, help='output FP32 .npz on CUDA host, input on FPGA host')
    ap.add_argument('--out', default=None, help='board result .npz')
    ap.add_argument('--dev', default='/dev/xdma0')
    ap.add_argument('--device-rows', action='store_true')
    ap.add_argument('--check-isa', action='store_true')
    ap.add_argument('--wsplit', action='store_true')
    ap.add_argument('--host-attention', action='store_true')
    ap.add_argument('--strict-fpga-only', action='store_true',
                    help='refuse this runner: its graph still contains CPU numerical operations')
    ap.add_argument('--max-rel-l2', type=float, default=None,
                    help='fail board comparison above this explicit relative-L2 threshold')
    args = ap.parse_args()

    if args.strict_fpga_only:
        ap.error('strict FPGA-only execution is not implemented by this hybrid SmolVLA runner')
    if args.max_rel_l2 is not None and (not np.isfinite(args.max_rel_l2) or args.max_rel_l2 < 0):
        ap.error('--max-rel-l2 must be finite and nonnegative')

    import torch
    # Deterministic CPU input tensors on BOTH hosts, irrespective of CUDA availability.
    torch.manual_seed(args.seed)
    pol, cfg = sm.real_policy(args.policy)
    if args.steps is not None:
        cfg.num_steps = pol.model.config.num_steps = args.steps
    vocab = pol.model.vlm_with_expert.get_vlm_model().text_model.embed_tokens.num_embeddings
    x = sm.inputs(cfg, args.seed, vocab)
    in_hash = input_fingerprint(x)
    w_hash = weight_fingerprint(pol)
    nparam = sum(p.numel() for p in pol.parameters())
    d = cfg.action_feature.shape[0] if hasattr(cfg, 'action_feature') else cfg.max_action_dim
    print(f'Policy {args.policy} | {nparam/1e6:.1f}M params | seed {args.seed} | steps {cfg.num_steps}', flush=True)
    print(f'Input fingerprint: {in_hash}\nWeight sample fingerprint: {w_hash}', flush=True)

    if args.mode == 'reference':
        if not torch.cuda.is_available():
            raise SystemExit('CUDA is unavailable in this Python environment. Check torch CUDA build and nvidia-smi.')
        print('CUDA device:', torch.cuda.get_device_name(0), flush=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        pol.to('cuda')
        gx = to_device(x, 'cuda')
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            y = pol.model.sample_actions(*gx[:5], noise=gx[5].clone())
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        ref = y.detach().float().cpu().numpy()[0, :, :d].copy()
        if not np.isfinite(ref).all():
            raise RuntimeError('Reference contains non-finite values')
        Path(args.ref).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.ref, ref=ref,
                            seed=np.int64(args.seed), steps=np.int64(cfg.num_steps),
                            policy=args.policy, input_sha256=in_hash,
                            weight_sample_sha256=w_hash,
                            pytorch_version=torch.__version__)
        print(f'CUDA reference saved: {args.ref}, shape={ref.shape}, time={elapsed:.2f}s', flush=True)
        return 0

    with np.load(args.ref, allow_pickle=False) as z:
        for key, expected in [('seed', args.seed), ('steps', cfg.num_steps),
                              ('policy', args.policy), ('input_sha256', in_hash),
                              ('weight_sample_sha256', w_hash)]:
            got = take_scalar(z, key)
            if str(got) != str(expected):
                raise SystemExit(f'Reference mismatch {key}: file={got!r}, current={expected!r}. '
                                 'Use identical checkpoint, seed, steps and versions.')
        ref = np.asarray(z['ref'], dtype=np.float32).copy()

    from opentpu import torchdev as TD
    # No FP32 CPU reference and no deepcopy of the model in board mode.
    runner = TD.Runner('board', dev=args.dev, check_isa=args.check_isa, resident=True)
    cov = TD.Coverage()
    n = TD.to_device(pol, runner, cov, wsplit=args.wsplit, rows=args.device_rows)
    vwe = pol.model.vlm_with_expert
    if not args.host_attention:
        att = TD.DeviceAttention(runner, cov, vwe.num_attention_heads, vwe.num_key_value_heads)
        vwe.eager_attention_forward = att
        TD.hf_attention(vwe.vlm.model.vision_model, att)
    print(f'FPGA backend: {n} converted modules, MCOLS={runner.cfg.MCOLS}, '
          f'--device-rows={args.device_rows}, --check-isa={args.check_isa}', flush=True)
    ops = TD.device_ops(runner, cov) if args.device_rows else contextlib.nullcontext()
    t0 = time.perf_counter()
    with torch.no_grad(), TD.host_counter(cov), ops:
        y = pol.model.sample_actions(*x[:5], noise=x[5].clone())
    elapsed = time.perf_counter() - t0
    got = y.detach().float().cpu().numpy()[0, :, :d].copy()
    if got.shape != ref.shape:
        raise RuntimeError(f'Shape mismatch FPGA {got.shape} / reference {ref.shape}')
    if not np.isfinite(got).all():
        raise RuntimeError('FPGA result contains non-finite values')
    diff = got.astype(np.float64) - ref.astype(np.float64)
    den = float(np.linalg.norm(ref.astype(np.float64)))
    rel = float(np.linalg.norm(diff)/den) if den else float(np.linalg.norm(diff))
    max_abs = float(np.abs(diff).max(initial=0))
    print(cov.report(), flush=True)
    print(f'FPGA result: rel_L2={rel:.8f}, max_abs={max_abs:.8f}, '
          f'kernel_runs={runner.stats["runs"]}, fpga_cycles={runner.cycles:,}, '
          f'fpga_backend_elapsed={runner.device_s:.2f}s, full_wall={elapsed:.2f}s', flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, ref=ref, dev=got,
                            rel_l2=rel, max_abs=max_abs,
                            input_sha256=in_hash,
                            weight_sample_sha256=w_hash)
        print('Saved board comparison:', args.out, flush=True)
    if args.max_rel_l2 is not None and rel > args.max_rel_l2:
        raise SystemExit(f'FAIL: rel_L2 {rel:.8f} exceeds {args.max_rel_l2:.8f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
