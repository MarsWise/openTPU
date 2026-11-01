"""Diffusion Policy (LeRobot, Apache-2.0; Chi et al. 2023) on openTPU: the policy's own PyTorch
inference (DiffusionModel.generate_actions: the ResNet-18 image encoder with GroupNorm and
spatial softmax, then the DDPM denoising loop over the conditional 1-D U-Net) with every
nn.Conv2d (the ResNet), nn.Conv1d / nn.ConvTranspose1d (the U-Net: im2col / col2im on the host,
the GEMM on the device) and nn.Linear (FiLM, timestep MLP, encoder head) on the device
(opentpu/torchdev.py: int8 MX weights, QACT activations) on the ISA simulator or the card.
With --device-ops also GroupNorm, Mish, ReLU, the spatial soft-argmax, the DDPM update (its
scalar coefficients and the noise draw on the host) and the sums and products of fp32 tensors
run on the device (torchdev DeviceGroupNorm / DeviceRows / DeviceSpatialSoftmax / device_ddpm /
device_ops); without it they run in PyTorch on the host. The max-pool, im2col / col2im and layout
always do. Everything on the host is counted (Coverage). Actions are compared with the same policy in float32 from the same
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


def pusht_shaped(seed: int = 0):
    """A random policy of lerobot/diffusion_pusht's shapes (offline: 96x96 image cropped to 84,
    2-D state and actions, horizon 16, U-Net 512 / 1,024 / 2,048, ResNet-18 with GroupNorm)."""
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    cfg = DiffusionConfig(n_obs_steps=2, horizon=16, n_action_steps=8, crop_shape=(84, 84),
                          crop_is_random=False, pretrained_backbone_weights=None,
                          use_group_norm=True, spatial_softmax_num_keypoints=32,
                          num_inference_steps=100)
    cfg.input_features = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,))}
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
    torch.manual_seed(seed)
    return DiffusionPolicy(cfg).float().eval(), cfg


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
    ap.add_argument("--pusht-shaped", action="store_true",
                    help="a random policy of diffusion_pusht's shapes (offline)")
    ap.add_argument("--card-config", action="store_true",
                    help="--backend isa with the r5-m8 card's configuration (MCOLS 8, 128K TMEM, "
                         "PAIR / TPAIR / TQUAD / TFN / DSTEP / STREAM)")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--backend", default="isa", choices=["isa", "board"])
    ap.add_argument("--check-isa", action="store_true")
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--wsplit", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None)
    ap.add_argument("--resident", action="store_true",
                    help="the whole policy as a resident graph (opentpu/diffusion_resident.py): "
                         "encoders, U-Net, DDPM updates on the device, the tensors in its DRAM; "
                         "the host writes the inputs and reads the actions")
    ap.add_argument("--device-ops", action="store_true",
                    help="GroupNorm, Mish, ReLU, spatial softmax and large adds / products on "
                         "the device too")
    ap.add_argument("--device-noise", action="store_true",
                    help="with --resident: the initial and per-step noise drawn on the device "
                         "from --seed (VOP RAND, a VRAND bitstream); the reference is the "
                         "policy's float loop given those noises")
    ap.add_argument("--strict-fpga-only", action="store_true",
                    help="reject this hybrid runner until all numerical operations are on the FPGA")
    a = ap.parse_args()
    if a.strict_fpga_only:
        ap.error("strict FPGA-only execution is unavailable: this runner computes max-pool, "
                 "im2col / col2im, layout and the scheduler's coefficients and noise on the host "
                 "(and GroupNorm, Mish, spatial softmax, the DDPM update without --device-ops)")
    if not (a.tiny or a.policy or a.pusht_shaped):
        ap.error("--tiny, --pusht-shaped or --policy")
    import torch
    from opentpu import torchdev as TD
    pol, cfg = (tiny_policy(a.seed) if a.tiny else pusht_shaped(a.seed) if a.pusht_shaped
                else real_policy(a.policy))
    dm = pol.diffusion
    if a.steps:
        dm.num_inference_steps = a.steps
    nparam = sum(p.numel() for p in pol.parameters())
    what = ("tiny (random, offline)" if a.tiny else "pusht-shaped (random, offline)"
            if a.pusht_shaped else a.policy)
    print(f"Diffusion Policy {what}: "
          f"{nparam / 1e6:.1f} M params, horizon {cfg.horizon}, {dm.num_inference_steps} "
          f"{cfg.noise_scheduler_type} steps, U-Net {tuple(cfg.down_dims)}, "
          f"{cfg.vision_backbone}")
    b, noise = inputs(cfg, a.seed)
    if a.resident:
        return run_resident(a, pol, cfg, b, noise)
    with torch.no_grad():
        torch.manual_seed(a.seed + 1)           # the scheduler's per-step noise
        t0 = time.time()
        ref = dm.generate_actions(b, noise=noise.clone())
        t_ref = time.time() - t0
    dev = copy.deepcopy(pol)
    rcfg = None
    if a.card_config and a.backend == "isa":
        from opentpu.isasim import board_config
        rcfg = board_config(MCOLS=8, TMEM_WORDS=131072, PAIR=True, TPAIR=True, TQUAD=True,
                            TFN=True, DSTEP=True, STREAM=True, DRAM_BYTES=1 << 30)
    runner = TD.Runner(a.backend, cfg=rcfg, dev=a.dev, check_isa=a.check_isa)
    cov = TD.Coverage()
    n = TD.to_device(dev, runner, cov, wsplit=a.wsplit, rows=a.device_ops)
    if a.device_ops and not TD.device_ddpm(dev.diffusion.noise_scheduler, runner, cov,
                                                steps=dev.diffusion.num_inference_steps):
        print("the scheduler's update stays on the host (not DDPM epsilon / fixed_small)")
    print(f"{n} modules (nn.Linear / nn.Conv2d / nn.Conv1d / nn.ConvTranspose1d) on the device "
          f"({a.backend}, D={runner.cfg.D}, MCOLS={runner.cfg.MCOLS}"
          f"{', wsplit' if a.wsplit else ''})")
    ops = TD.device_ops(runner, cov, min_elems=1) if a.device_ops else contextlib.nullcontext()
    with torch.no_grad(), TD.host_counter(cov), ops:
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


def run_resident(a, pol, cfg, b, noise):
    """The policy on the device with nothing computed on the host: the observation (normalized
    images and states), the initial noise and the scheduler's per-step noise written once, every
    step's programs launched, the actions read back. Compared with the policy's float32 loop."""
    import torch
    from opentpu import diffusion_resident as DR, resident as RG
    from opentpu.isasim import board_config
    dm = pol.diffusion
    steps = dm.num_inference_steps
    with torch.no_grad():
        gc = dm._prepare_global_conditioning(b)[0].numpy()     # (the float reference's)
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
    loop = DR.build(pol, c, steps, encode=True, device_noise=dn)
    progs = loop.programs()
    t_c = time.time() - t0
    t0 = time.time()
    if dn:                                   # the device's noises, the float loop given them
        z0, zs = loop.noises(a.seed + 1)
        ref = DR.reference_given(pol, gc, z0, zs, steps)
    else:
        ref, zs = DR.reference_noise(pol, gc, noise[0].numpy(), steps, seed=a.seed + 1)
    t_ref = time.time() - t0
    kw = dict(images=b["observation.images"][0, :, 0].numpy(),
              states=b["observation.state"][0].numpy())
    img = loop.image(None, None, None, seed=a.seed + 1, **kw) if dn else \
        loop.image(None, noise[0].numpy(), zs, **kw)
    print(f"resident policy: {len(progs)} programs ({len(loop.enc_progs)} encoder, "
          f"{len(loop.common) + 2} a step x {steps}), DRAM image {len(img) / 2**20:.1f} MiB, "
          f"compiled in {t_c:.1f}s")
    t0 = time.time()
    if board is None:
        img, n = RG.run_isa(c, progs, img)
        got = loop.actions(img)
        st = {"instructions": n}
    else:
        board.write(0, img)
        at = -(-(len(img) + (1 << 20)) // 4096) * 4096
        st = RG.run_board(board, progs, at)
        got = board.read(loop.sample.addr, loop.sample.nbytes).view(np.float32)
        got = got.reshape(-1, loop.A)[loop.sample.pad:loop.sample.pad + loop.T]
        if a.check_isa:
            iso, _ = RG.run_isa(c, progs, img)
            print("card == ISA:", np.array_equal(got, loop.actions(iso)))
    t_dev = time.time() - t0
    S = dm.config.n_obs_steps
    r = ref[S - 1:S - 1 + dm.config.n_action_steps]
    g = got[S - 1:S - 1 + dm.config.n_action_steps]
    rel = float(np.linalg.norm(g - r) / np.linalg.norm(r))
    print(f"device run {t_dev:.1f}s ({st}), float run {t_ref:.2f}s")
    print(f"actions [{r.shape[0]} x {r.shape[1]}] vs float32: rel L2 {rel:.4f}, max |d| "
          f"{float(np.abs(g - r).max()):.4f}; host: inputs written, actions read, nothing "
          f"computed{' (the noise drawn on the device from the seed)' if dn else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
