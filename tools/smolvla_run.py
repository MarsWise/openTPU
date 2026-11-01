"""SmolVLA (LeRobot, Apache-2.0) on openTPU: the policy's own PyTorch inference
(VLAFlowMatching.sample_actions: SigLIP vision tower, connector, the VLM prefix with its KV
cache, the action expert's 10 Euler steps) with every nn.Linear on the device
and the SigLIP patch embedding (a 16x16 stride-16 conv) on the device (opentpu/torchdev.py,
kernels/linear.py and kernels/conv.py: int8 MX weights, QACT activations) on the ISA simulator
or the card; attention products, norms, RoPE, softmax, the pixel shuffle and the Euler update
run in PyTorch on the host and are counted (Coverage). Actions are
compared with the same policy in float32 from the same inputs and noise.

Usage:
  python tools/smolvla_run.py --tiny [--backend isa]            # a random tiny SmolVLA, offline
  python tools/smolvla_run.py --policy lerobot/smolvla_base [--backend board --check-isa]
        [--wsplit] [--steps 10] [--seed 0] [--episode DATASET_DIR]
--policy: a LeRobot SmolVLA checkpoint (directory or HF id; its VLM's config and processor come
from its vlm_model_name). Inputs: a deterministic synthetic observation (image, state, language
tokens) unless --episode gives recorded frames (not yet implemented: the host plan's step).
No robot is controlled: the actions are only compared.
"""
import argparse
import copy
import sys
import time

import numpy as np


def tiny_policy(seed: int = 0):
    """A random SmolVLA built offline: SmolVLM's architecture at toy sizes (SigLIP 2 layers,
    64x64 images of 16x16 patches, a 4-layer Llama text model), the expert 0.75 wide,
    cross-attention with self-attention every 2nd layer, as the defaults."""
    import torch
    import transformers
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla import smolvlm_with_expert as SE
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    vc = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2,
              image_size=64, patch_size=16)
    tc = dict(model_type="llama", hidden_size=128, intermediate_size=256, num_hidden_layers=4,
              num_attention_heads=4, num_key_value_heads=2, vocab_size=1000,
              max_position_embeddings=512, rms_norm_eps=1e-5, head_dim=32)
    hf = transformers.SmolVLMConfig(vision_config=vc, text_config=tc, scale_factor=2,
                                    image_token_id=999)
    SE.AutoConfig = type("C", (), {"from_pretrained": staticmethod(lambda *_a, **_k: hf)})
    from types import SimpleNamespace
    proc = SimpleNamespace(tokenizer=SimpleNamespace(fake_image_token_id=997,
                                                     global_image_token_id=998))
    SE.AutoProcessor = type("P", (), {"from_pretrained": staticmethod(lambda *_a, **_k: proc)})
    cfg = SmolVLAConfig(vlm_model_name="tiny-offline", load_vlm_weights=False,
                        num_vlm_layers=4, chunk_size=8, n_action_steps=8,
                        resize_imgs_with_padding=(64, 64), num_steps=10)
    cfg.input_features = {
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,))}
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
    torch.manual_seed(seed)
    pol = SmolVLAPolicy(cfg).float().eval()
    with torch.no_grad():                   # random weights at a scale whose activations matter
        for n, p in pol.named_parameters():
            if p.dim() >= 2:
                p.normal_(0, 1.0 / np.sqrt(p.shape[-1]))
            elif "norm" in n:
                p.uniform_(0.8, 1.2)
    return pol, cfg


def real_policy(path: str):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    pol = SmolVLAPolicy.from_pretrained(path).float().eval()
    return pol, pol.config


def inputs(cfg, seed: int, vocab: int):
    """(images, img_masks, lang_tokens, lang_masks, state, noise): one camera image in [-1, 1]
    at the policy's resolution, a state in the normalized range, 12 language tokens, noise."""
    import torch
    g = torch.Generator().manual_seed(seed)
    W, H = cfg.resize_imgs_with_padding
    imgs, masks = [], []
    for _ in cfg.image_features:
        imgs.append(torch.rand(1, 3, H, W, generator=g) * 2 - 1)
        masks.append(torch.ones(1, dtype=torch.bool))
    toks = torch.randint(1, min(vocab, 30000), (1, 12), generator=g)
    lmask = torch.ones(1, 12, dtype=torch.bool)
    state = torch.randn(1, cfg.max_state_dim, generator=g)
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=g)
    return imgs, masks, toks, lmask, state, noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--backend", default="isa", choices=("isa", "board"))
    ap.add_argument("--check-isa", action="store_true")
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--wsplit", action="store_true")
    ap.add_argument("--device-rows", action="store_true",
                    help="activations, LayerNorm / RMSNorm and tensor sums / products on the "
                         "device too (kernels/eltwise.py)")
    ap.add_argument("--no-resident", action="store_true",
                    help="send each layer's weights with every call (the old way)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--card-config", action="store_true",
                    help="--backend isa with the r5-m8 card's configuration (TFN etc.)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None)
    ap.add_argument("--resident", action="store_true",
                    help="the whole policy as a resident graph (opentpu/smolvla_resident.py): "
                         "vision, connector, prefix layers and every Euler step of the expert "
                         "on the device; the host writes the preprocessed images, state, "
                         "noise and the instruction's token rows, and reads the actions")
    ap.add_argument("--device-noise", action="store_true",
                    help="with --resident: the flow's initial noise drawn on the device from "
                         "--seed (VOP RAND, a VRAND bitstream); the reference is sample_actions "
                         "given that noise")
    ap.add_argument("--host-attention", action="store_true",
                    help="the attention products on the host (stage 1)")
    a = ap.parse_args()
    if not (a.tiny or a.policy):
        ap.error("--tiny or --policy")
    import torch
    from opentpu import torchdev as TD
    pol, cfg = tiny_policy(a.seed) if a.tiny else real_policy(a.policy)
    if a.steps:
        cfg.num_steps = pol.model.config.num_steps = a.steps
    nparam = sum(p.numel() for p in pol.parameters())
    vocab = pol.model.vlm_with_expert.get_vlm_model().text_model.embed_tokens.num_embeddings
    print(f"SmolVLA {'tiny (random, offline)' if a.tiny else a.policy}: {nparam / 1e6:.1f} M "
          f"params, chunk {cfg.chunk_size}, {cfg.num_steps} Euler steps, "
          f"{cfg.num_vlm_layers} VLM layers")
    x = inputs(cfg, a.seed, vocab)
    with torch.no_grad():
        t0 = time.time()
        ref = pol.model.sample_actions(*x[:5], noise=x[5].clone())
        t_ref = time.time() - t0
    if a.resident:
        return run_resident(a, pol, x, ref[0].numpy(), t_ref)
    dev = copy.deepcopy(pol)
    rcfg = None
    if a.card_config and a.backend == "isa":     # the card's configuration (TFN: sin / cos)
        from opentpu.isasim import board_config
        rcfg = board_config(MCOLS=8, TMEM_WORDS=131072, PAIR=True, TPAIR=True, TQUAD=True,
                            TFN=True, DSTEP=True, STREAM=True, DRAM_BYTES=1 << 30)
    runner = TD.Runner(a.backend, cfg=rcfg, dev=a.dev, check_isa=a.check_isa,
                       resident=not a.no_resident)
    cov = TD.Coverage()
    n = TD.to_device(dev, runner, cov, wsplit=a.wsplit, rows=a.device_rows)
    vwe = dev.model.vlm_with_expert
    if not a.host_attention:                # SmolVLA's attention (its eager path) on the device
        att = TD.DeviceAttention(runner, cov, vwe.num_attention_heads, vwe.num_key_value_heads)
        vwe.eager_attention_forward = att           # the VLM / expert layers (prefix masks)
        TD.hf_attention(vwe.vlm.model.vision_model, att)    # the SigLIP tower (no mask)
    print(f"{n} modules (nn.Linear / nn.Conv2d{' / activations / norms' if a.device_rows else ''}) on the device ({a.backend}, D={runner.cfg.D}, "
          f"MCOLS={runner.cfg.MCOLS}{', wsplit' if a.wsplit else ''})")
    import contextlib
    ops = TD.device_ops(runner, cov, min_elems=1) if a.device_rows else contextlib.nullcontext()
    with torch.no_grad(), TD.host_counter(cov), ops:
        t0 = time.time()
        got = dev.model.sample_actions(*x[:5], noise=x[5].clone())
        t_dev = time.time() - t0
    r, g = ref.numpy()[0], got.numpy()[0]
    d = cfg.action_feature.shape[0] if hasattr(cfg, "action_feature") else r.shape[-1]
    r, g = r[:, :d], g[:, :d]
    rel = float(np.linalg.norm(g - r) / np.linalg.norm(r))
    print(cov.report())
    print(f"kernel runs {runner.stats['runs']}, device time {runner.device_s:.1f}s"
          + (f", card cycles {runner.cycles:,}" if runner.board is not None else "")
          + f"; float run {t_ref:.2f}s, device run {t_dev:.1f}s")
    print(f"bytes to the device: weights {runner.weight_bytes:,} once (resident), per call "
          f"{runner.call_bytes:,} in all (activations, programs"
          + ("" if runner.resident else ", weights") + ")")
    print(f"programs traced {runner.stats['traced']}, reused {runner.stats['cached']}")
    print(f"actions [{r.shape[0]} x {d}] vs float32: rel L2 {rel:.4f}, max |d| "
          f"{float(np.abs(g - r).max()):.4f} (action range {float(np.abs(r).max()):.3f}); "
          f"first step float {np.round(r[0], 3).tolist()} device {np.round(g[0], 3).tolist()}")
    if a.save:
        np.savez(a.save, ref=r, dev=g)
    return 0


def run_resident(a, pol, x, ref, t_ref):
    """SmolVLA with nothing computed on the host (the inputs written once, the actions read)."""
    from opentpu import resident as RG, smolvla_resident as SR
    from opentpu.isasim import board_config
    imgs, masks, toks, lmask, state, noise = x
    if not (all(bool(mk.all()) for mk in masks) and bool(lmask.all())):
        raise SystemExit("--resident takes every image and language token valid (no padding)")
    board = None
    if a.backend == "board":
        from opentpu.host.board import Board, XdmaTransport, device_config
        board = Board(XdmaTransport(a.dev), check=False)
        c = device_config(board.info())
    else:
        c = board_config(MCOLS=8, TMEM_WORDS=131072, PAIR=True, TPAIR=True, TQUAD=True,
                         TFN=True, DSTEP=True, STREAM=True, DRAM_BYTES=1 << 30)
    dn = getattr(a, "device_noise", False)
    if dn and not getattr(c, "VRAND", False):
        if board is not None:
            raise SystemExit("--device-noise: the bitstream has no VOP RAND (CAPS2 bit10)")
        import dataclasses
        c = dataclasses.replace(c, VRAND=True)
    t0 = time.time()
    vla = SR.build(pol, c, toks[0].numpy(), device_noise=dn)
    progs = vla.programs()
    pix = [i[0].numpy() for i in imgs]
    if dn:                                   # the device's noise; the float run given it
        import torch
        z = torch.as_tensor(vla.noise(a.seed + 1))[None]
        t1 = time.time()
        with torch.no_grad():
            ref = pol.model.sample_actions(imgs, masks, toks, lmask, state,
                                           noise=z.clone())[0].numpy()
        t_ref = time.time() - t1
        img = vla.image(pix, state[0].numpy(), None, seed=a.seed + 1)
    else:
        img = vla.image(pix, state[0].numpy(), noise[0].numpy())
    print(f"resident policy: {len(progs)} programs ({len(vla.enc)} prefix, {len(vla.step[0])} "
          f"a step x {vla.steps}), DRAM image {len(img) / 2**20:.1f} MiB, compiled in "
          f"{time.time() - t0:.1f}s")
    t0 = time.time()
    if board is None:
        img, n = RG.run_isa(c, progs, img)
        got, st = vla.actions(img), {"instructions": n}
    else:
        board.write(0, img)
        st = RG.run_board(board, progs, -(-(len(img) + (1 << 20)) // 4096) * 4096)
        got = board.read(vla.x.addr, vla.x.nbytes).view(np.float32).reshape(vla.chunk, vla.A)
        if a.check_isa:
            iso, _ = RG.run_isa(c, progs, img)
            print("card == ISA:", np.array_equal(got, vla.actions(iso)))
    rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    print(f"device run {time.time() - t0:.1f}s ({st}), float run {t_ref:.2f}s")
    print(f"actions [{ref.shape[0]} x {ref.shape[1]}] vs float32: rel L2 {rel:.4f}, max |d| "
          f"{float(np.abs(got - ref).max()):.4f} (range {float(np.abs(ref).max()):.3f}); host: "
          f"inputs written, actions read, nothing computed"
          f"{' (the noise drawn on the device from the seed)' if dn else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
