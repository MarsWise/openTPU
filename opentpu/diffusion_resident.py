"""LeRobot's Diffusion Policy denoising loop as a resident graph (opentpu/resident.py): the 1-D
conditional U-Net (conv / GroupNorm / Mish blocks with FiLM, down and up sampling, skip
concatenations), the timestep embedding and its MLP, the condition encoders and the DDPM update
of every step run on the device over tensors that stay in the card's DRAM. The host writes the
inputs once -- the global condition (the observation encoder's output), the initial noise, the
timesteps and the scheduler's per-step variance noise (drawn as diffusers draws it: inputs, as a
seed is) -- launches the programs and reads the final actions.

build(policy, cfg) -> Loop; Loop.run_isa(...) / Loop.run_board(...).
"""
from __future__ import annotations

import math

import numpy as np

from . import resident as RG
from .compiler import Tensor


def _np(t):
    return t.detach().float().cpu().numpy()


class Loop:
    def __init__(self, g, sample, gfeat, times, noises, step_progs, common, coeffs, A, T, E):
        self.g, self.sample, self.gfeat, self.times = g, sample, gfeat, times
        self.noises, self.step_progs, self.common = noises, step_progs, common
        self.coeffs, self.A, self.T, self.E = coeffs, A, T, E

    def programs(self) -> list:
        """Every program of the loop in order: per step its timestep embedding, the shared
        U-Net, its DDPM update."""
        out = []
        for pre, post in self.step_progs:
            out += pre + self.common + post
        return out

    def image(self, global_cond, noise0, step_noise) -> np.ndarray:
        g = self.g
        img = g.image()
        g.put(img, self.gfeat.cols(self.E, self.gfeat.C), np.asarray(global_cond).reshape(1, -1))
        g.put(img, self.sample, np.asarray(noise0).reshape(self.T, self.A))
        g.put(img, self.times, np.asarray([c["t"] for c in self.coeffs], np.float32)[:, None])
        for nt, z in zip(self.noises, step_noise):
            g.put(img, nt, np.asarray(z).reshape(self.T, self.A))
        return img

    def actions(self, img) -> np.ndarray:
        return self.g.read(img, self.sample)


def build(policy, cfg, steps: int | None = None) -> Loop:
    """The resident loop of `policy` (a LeRobot DiffusionPolicy, B = 1)."""
    from .kernels import eltwise as E
    from .kernels.ddpm import ddpm_epsilon_rows
    dm = policy.diffusion
    un = dm.unet
    sch = dm.noise_scheduler
    c = sch.config
    if type(sch).__name__ != "DDPMScheduler" or c.prediction_type != "epsilon" or \
            c.variance_type != "fixed_small" or getattr(c, "thresholding", False):
        raise NotImplementedError("the resident loop takes DDPM epsilon / fixed_small")
    steps = steps or dm.num_inference_steps
    sch.set_timesteps(steps)
    T, A = dm.config.horizon, un.final_conv[1].out_channels
    g = RG.Graph(cfg)

    # inputs
    se = un.diffusion_step_encoder
    Edim = se[0].dim
    gcd = un.down_modules[0][0].cond_encoder[1].in_features - Edim
    sample = g.tensor(T, A, 2)
    gfeat = g.tensor(1, Edim + gcd)
    times = g.tensor(steps, 1)
    noises = [g.tensor(T, A) for _ in range(steps)]

    # per step: the timestep's sinusoidal embedding (TFN sin / cos of the input timestep)
    half = Edim // 2
    freq = g.vec(np.exp(np.arange(half) * -(math.log(10000) / (half - 1))))
    temb = g.tensor(1, 2 * half)
    pre_ops = [[("temb", E.sincos_embed, dict(x=times.rows(i, i + 1), f=freq, y=temb.rows(),
                                               M=1, N=2 * half))] for i in range(steps)]
    # shared: the timestep MLP into the global feature, the condition's Mish
    g.ops = []
    h = g.linear(temb, _np(se[1].weight), _np(se[1].bias), act="mish", name="tmlp1")
    g.linear(h, _np(se[3].weight), _np(se[3].bias), y=gfeat.cols(0, Edim), name="tmlp2")
    cm = g.unary(gfeat, "mish", name="cond_mish")

    def block(x, blk, name):
        c1, c2 = blk.conv1.block, blk.conv2.block
        k = c1[0].kernel_size[0]
        y = g.conv1d(x, _np(c1[0].weight), _np(c1[0].bias), 1, k // 2, name=name + ".conv1")
        g.group_norm(y, c1[1].num_groups, _np(c1[1].weight), _np(c1[1].bias), c1[1].eps,
                     act="mish", y=y, name=name + ".gn1")
        lin = blk.cond_encoder[1]
        e = g.linear(cm, _np(lin.weight), _np(lin.bias), name=name + ".cond")
        Co = blk.out_channels
        if blk.use_film_scale_modulation:
            g.film(y, Tensor(e.addr, (Co,), (1,)), Tensor(e.addr + 4 * Co, (Co,), (1,)),
                   name=name + ".film")
        else:
            g.op(name + ".cond_add", RG.binary_k, a=y.rows(), b=Tensor(e.addr, (Co,), (1,)),
                 y=y.rows(), op="add", brow=True)
        z = g.conv1d(y, _np(c2[0].weight), _np(c2[0].bias), 1, k // 2, name=name + ".conv2")
        g.group_norm(z, c2[1].num_groups, _np(c2[1].weight), _np(c2[1].bias), c2[1].eps,
                     act="mish", y=z, name=name + ".gn2")
        rc = blk.residual_conv
        r = x if not hasattr(rc, "weight") else \
            g.conv1d(x, _np(rc.weight), _np(rc.bias), 1, 0, name=name + ".res")
        return g.add(z, r, name=name + ".add")

    x = sample
    skips = []
    for i, (r1, r2, down) in enumerate(un.down_modules):
        x = block(x, r1, f"down{i}.0")
        x = block(x, r2, f"down{i}.1")
        skips.append(x)
        if hasattr(down, "weight"):
            x = g.conv1d(x, _np(down.weight), _np(down.bias), down.stride[0], down.padding[0],
                         name=f"down{i}.sample")
    for i, m in enumerate(un.mid_modules):
        x = block(x, m, f"mid{i}")
    for i, (r1, r2, up) in enumerate(un.up_modules):
        s = skips.pop()
        cat = g.tensor(x.L, x.C + s.C, 2)
        g.copy(x, cat.cols(0, x.C), name=f"up{i}.cat0")
        g.copy(s, cat.cols(x.C, x.C + s.C), name=f"up{i}.cat1")
        x = block(cat, r1, f"up{i}.0")
        x = block(x, r2, f"up{i}.1")
        if hasattr(up, "weight"):
            x = g.conv_transpose1d(x, _np(up.weight), _np(up.bias), up.stride[0],
                                   up.padding[0], name=f"up{i}.sample")
    fb, fc = un.final_conv[0].block, un.final_conv[1]
    k = fb[0].kernel_size[0]
    y = g.conv1d(x, _np(fb[0].weight), _np(fb[0].bias), 1, k // 2, name="final.conv")
    g.group_norm(y, fb[1].num_groups, _np(fb[1].weight), _np(fb[1].bias), fb[1].eps,
                 act="mish", y=y, name="final.gn")
    eps = g.conv1d(y, _np(fc.weight), _np(fc.bias), 1, 0, name="final.out")
    common_ops = g.ops

    # per step: the DDPM update of the sample (the timestep's coefficients compile in)
    coeffs, post_ops = [], []
    for i, t in enumerate(sch.timesteps.tolist()):
        t = int(t)
        prev = sch.previous_timestep(t)
        a = sch.alphas_cumprod[t]
        ap = sch.alphas_cumprod[prev] if prev >= 0 else sch.one
        beta, cur = 1 - a, a / ap
        k_ = dict(inv_sqrt_alpha_bar=float(1 / a.sqrt()),
                  sqrt_one_minus_alpha_bar=float(beta.sqrt()),
                  mean_x0=float(ap.sqrt() * (1 - cur) / beta),
                  mean_xt=float(cur.sqrt() * (1 - ap) / beta),
                  noise_scale=float(sch._get_variance(t).sqrt()) if t > 0 else 0.0,
                  clip_range=float(c.clip_sample_range) if c.clip_sample else 3.0e38)
        coeffs.append(dict(t=float(t), **k_))
        post_ops.append([("ddpm", ddpm_epsilon_rows,
                          dict(sample=sample.rows(), epsilon=eps.rows(), noise=noises[i].rows(),
                               y=sample.rows(), M=T, N=A, **k_))])
    common_progs = g.compile(common_ops)
    step_progs = [(g.compile(p), g.compile(q)) for p, q in zip(pre_ops, post_ops)]
    return Loop(g, sample, gfeat, times, noises, step_progs, common_progs, coeffs, A, T, Edim)


def reference_noise(policy, global_cond, noise0, steps: int, seed: int):
    """The policy's own float32 loop from global_cond and noise0 with torch seeded `seed` for
    the scheduler: (actions [T, A], the variance noise each step drew)."""
    import torch
    from diffusers.schedulers import scheduling_ddpm as SD
    dm = policy.diffusion
    dm.num_inference_steps = steps
    drawn = []
    orig = SD.randn_tensor

    def rec(*a, **k):
        z = orig(*a, **k)
        drawn.append(z.detach().clone())
        return z
    SD.randn_tensor = rec
    try:
        torch.manual_seed(seed)
        with torch.no_grad():
            out = dm.conditional_sample(1, global_cond=torch.as_tensor(global_cond)[None],
                                        noise=torch.as_tensor(noise0)[None].clone())
    finally:
        SD.randn_tensor = orig
    zs = [np.zeros_like(noise0)] * steps
    ts = [int(t) for t in dm.noise_scheduler.timesteps.tolist()]
    j = 0
    for i, t in enumerate(ts):                 # (diffusers draws only for t > 0)
        if t > 0:
            zs[i] = drawn[j][0].numpy()
            j += 1
    return out[0].numpy(), zs
