"""Diffusion Policy (LeRobot, Apache-2.0; Chi et al. 2023) on openTPU: the policy's own PyTorch
inference (DiffusionModel.generate_actions: the ResNet-18 image encoder with GroupNorm and
spatial softmax, then the DDPM denoising loop over the conditional 1-D U-Net) with every
nn.Conv2d (the ResNet), nn.Conv1d / nn.ConvTranspose1d (the U-Net: im2col / col2im on the host,
the GEMM on the device) and nn.Linear (FiLM, timestep MLP, encoder head) on the device
(opentpu/torchdev.py: int8 MX weights, QACT activations) on the ISA simulator or the card.
GroupNorm, Mish, spatial softmax and the scheduler's update run in PyTorch on the host and are
counted (Coverage). Actions are compared with the same policy in float32 from the same
observation, initial noise and scheduler noise (torch seeded identically for both runs).

Usage:
  python tools/diffusion_run.py --tiny [--backend isa]               # a random small policy
  python tools/diffusion_run.py --policy lerobot/diffusion_pusht [--backend board]
        [--steps 10] [--seed 0] [--check-isa]
--steps: denoising steps (default: the checkpoint's, 100 for diffusion_pusht's DDPM). The
observation is synthetic (a fixed random image and state, normalized). No robot is controlled.
"""
import argparse
import contextlib
import copy
import sys
import time

import numpy as np


def tiny_policy(seed: int = 0):
    """A random Diffusion Policy built offline: pusht's layout at small sizes (a 32x32 image,
    2-D state and actions, horizon 16, U-Net dims 32 / 64 / 128, ResNet-18 with GroupNorm)."""
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    cfg = DiffusionConfig(n_obs_steps=2, horizon=16, n_action_steps=8, down_dims=(32, 64, 128),
                          diffusion_step_embed_dim=32, kernel_size=5, n_groups=8,
                          pretrained_backbone_weights=None, use_group_norm=True,
                          spatial_softmax_num_keypoints=8, crop_shape=None,
                          num_inference_steps=5)
    cfg.input_features = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,))}
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
    torch.manual_seed(seed)
    return DiffusionPolicy(cfg).float().eval(), cfg


def real_policy(name):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    pol = DiffusionPolicy.from_pretrained(name).float().eval()
    return pol, pol.config


def inputs(cfg, seed):
    """A normalized observation batch of one (state, images) and the initial noise."""
    import torch
    g = torch.Generator().manual_seed(seed)
    S = cfg.n_obs_steps
    state = torch.randn(1, S, cfg.robot_state_feature.shape[0], generator=g)
    imgs = [torch.rand(1, S, *ft.shape, generator=g) for ft in cfg.image_features.values()]
    b = {"observation.state": state, "observation.images": torch.stack(imgs, dim=2)}
    noise = torch.randn(1, cfg.horizon, cfg.action_feature.shape[0], generator=g)
    return b, noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--backend", default="isa", choices=["isa", "board"])
    ap.add_argument("--check-isa", action="store_true")
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--wsplit", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    if not (a.tiny or a.policy):
        ap.error("--tiny or --policy")
    import torch
    from opentpu import torchdev as TD
    pol, cfg = tiny_policy(a.seed) if a.tiny else real_policy(a.policy)
    dm = pol.diffusion
    if a.steps:
        dm.num_inference_steps = a.steps
    nparam = sum(p.numel() for p in pol.parameters())
    print(f"Diffusion Policy {'tiny (random, offline)' if a.tiny else a.policy}: "
          f"{nparam / 1e6:.1f} M params, horizon {cfg.horizon}, {dm.num_inference_steps} "
          f"{cfg.noise_scheduler_type} steps, U-Net {tuple(cfg.down_dims)}, "
          f"{cfg.vision_backbone}")
    b, noise = inputs(cfg, a.seed)
    with torch.no_grad():
        torch.manual_seed(a.seed + 1)           # the scheduler's per-step noise
        t0 = time.time()
        ref = dm.generate_actions(b, noise=noise.clone())
        t_ref = time.time() - t0
    dev = copy.deepcopy(pol)
    runner = TD.Runner(a.backend, dev=a.dev, check_isa=a.check_isa)
    cov = TD.Coverage()
    n = TD.to_device(dev, runner, cov, wsplit=a.wsplit)
    print(f"{n} modules (nn.Linear / nn.Conv2d / nn.Conv1d / nn.ConvTranspose1d) on the device "
          f"({a.backend}, D={runner.cfg.D}, MCOLS={runner.cfg.MCOLS}"
          f"{', wsplit' if a.wsplit else ''})")
    with torch.no_grad(), TD.host_counter(cov), contextlib.nullcontext():
        torch.manual_seed(a.seed + 1)
        t0 = time.time()
        got = dev.diffusion.generate_actions(b, noise=noise.clone())
        t_dev = time.time() - t0
    r, g = ref.numpy()[0], got.numpy()[0]
    rel = float(np.linalg.norm(g - r) / np.linalg.norm(r))
    print(cov.report())
    print(f"kernel runs {runner.stats['runs']}, device time {runner.device_s:.1f}s"
          + (f", card cycles {runner.cycles:,}" if runner.board is not None else "")
          + f"; float run {t_ref:.2f}s, device run {t_dev:.1f}s")
    print(f"bytes to the device: weights {runner.weight_bytes:,} once (resident), per call "
          f"{runner.call_bytes:,} in all")
    print(f"programs traced {runner.stats['traced']}, reused {runner.stats['cached']}")
    print(f"actions [{r.shape[0]} x {r.shape[1]}] vs float32: rel L2 {rel:.4f}, max |d| "
          f"{float(np.abs(g - r).max()):.4f} (action range {float(np.abs(r).max()):.3f}); "
          f"first step float {np.round(r[0], 3).tolist()} device {np.round(g[0], 3).tolist()}")
    if runner.mismatch:
        print(f"CARD != ISA in {runner.mismatch} runs")
    if a.save:
        np.savez(a.save, ref=r, dev=g)
    return 0


if __name__ == "__main__":
    sys.exit(main())
