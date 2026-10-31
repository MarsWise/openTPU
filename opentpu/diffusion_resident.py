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
        self.enc_progs, self.enc_in = [], []         # (with the observation encoders)

    def programs(self) -> list:
        """Every program of the loop in order: per step its timestep embedding, the shared
        U-Net, its DDPM update."""
        out = list(self.enc_progs)
        for pre, post in self.step_progs:
            out += pre + self.common + post
        return out

    def image(self, global_cond, noise0, step_noise, images=None, states=None) -> np.ndarray:
        """global_cond (no encoders), or the observation's images ([S][C, H, W], normalized)
        and states ([S][sd]): the encoders compute the rest of the global condition."""
        g = self.g
        img = g.image()
        if self.enc_in:
            f = img.view(np.float32)
            for (it, st), x, sv in zip(self.enc_in, images, states):
                x = np.asarray(x, np.float32).transpose(1, 2, 0).reshape(-1)  # (NHWC input)
                f[it.addr // 4:it.addr // 4 + x.size] = x
                g.put(img, st, np.asarray(sv, np.float32).reshape(1, -1))
        else:
            g.put(img, self.gfeat.cols(self.E, self.gfeat.C),
                  np.asarray(global_cond).reshape(1, -1))
        g.put(img, self.sample, np.asarray(noise0).reshape(self.T, self.A))
        g.put(img, self.times, np.asarray([c["t"] for c in self.coeffs], np.float32)[:, None])
        for nt, z in zip(self.noises, step_noise):
            g.put(img, nt, np.asarray(z).reshape(self.T, self.A))
        return img

    def actions(self, img) -> np.ndarray:
        return self.g.read(img, self.sample)


def build(policy, cfg, steps: int | None = None, encode: bool = False) -> Loop:
    """The resident loop of `policy` (a LeRobot DiffusionPolicy, B = 1); with `encode` also its
    observation encoders (images -> features, beside the states, into the global condition)."""
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

    enc_ops, enc_in = [], []
    if encode:
        dc = dm.config
        if dc.env_state_feature or len(dc.image_features) != 1:
            raise NotImplementedError("the resident encoder takes one camera, no env state")
        enc = dm.rgb_encoder
        enc = enc[0] if isinstance(enc, torch_modulelist()) else enc
        _, Hi, Wi = list(dc.image_features.values())[0].shape
        sd = dc.robot_state_feature.shape[0]
        F = enc.out.out_features
        for o in range(dc.n_obs_steps):
            c0 = Edim + o * (sd + F)
            st = gfeat.cols(c0, c0 + sd)
            it = encoder(g, enc, Hi, Wi, Tensor(gfeat.addr + 4 * (c0 + sd), (F,), (1,)),
                         name=f"obs{o}")
            enc_in.append((it, st))
        enc_ops, g.ops = g.ops, []

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
    loop = Loop(g, sample, gfeat, times, noises, step_progs, common_progs, coeffs, A, T, Edim)
    loop.enc_progs, loop.enc_in = g.compile(enc_ops), enc_in
    return loop


def torch_modulelist():
    import torch
    return torch.nn.ModuleList


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


def encoder(g, enc, H: int, W: int, out_vec: Tensor, name: str = "rgb"):
    """LeRobot's DiffusionRgbEncoder (ResNet with GroupNorm, SpatialSoftmax, linear + ReLU) as
    resident ops: returns the [H, W, 3] input image the host writes (NHWC, normalized as the
    policy's inputs are); the feature goes to out_vec ([F] words of a device tensor). A center
    crop is a device copy of the crop's rows."""
    from .compiler import Tensor as Tn
    D = g.D
    img = g.tensor2(H, W, 3)
    if getattr(enc, "do_crop", False):
        ch, cw = enc.center_crop.size if hasattr(enc.center_crop, "size") else enc.center_crop
        oy, ox = (H - ch) // 2, (W - cw) // 2
        x = g.tensor2(ch, cw, 3, 3, D)
        g.op(name + ".crop", RG.copy_k,
             x=Tn(img.addr + 4 * (oy * W * 3 + ox * 3), (ch, cw * 3), (W * 3, 1)),
             y=Tn(x.interior().base, (ch, cw * 3), (x.rs, 1)))
    else:
        x = g.tensor2(H, W, 3, 3, D)
        g.op(name + ".place", RG.copy_k, x=Tn(img.addr, (H, W * 3), (W * 3, 1)),
             y=Tn(x.interior().base, (H, W * 3), (x.rs, 1)))
    bb = list(enc.backbone)
    conv1, gn1, _relu, mp = bb[:4]
    y = g.conv2d(x, _np(conv1.weight), None if conv1.bias is None else _np(conv1.bias),
                 conv1.stride[0], conv1.padding[0], name=name + ".conv1")
    y = g.group_norm2d(y, gn1.num_groups, _np(gn1.weight), _np(gn1.bias), gn1.eps, "relu",
                       border=mp.padding, name=name + ".gn1")
    k = mp.kernel_size if isinstance(mp.kernel_size, int) else mp.kernel_size[0]
    s = mp.stride if isinstance(mp.stride, int) else mp.stride[0]
    x = g.maxpool2d(y, k, s, mp.padding, name=name + ".maxpool")
    for li, layer in enumerate(bb[4:]):
        for bi, blk in enumerate(layer):
            n = f"{name}.l{li + 1}.{bi}"
            h = g.conv2d(x, _np(blk.conv1.weight), None, blk.conv1.stride[0],
                         blk.conv1.padding[0], name=n + ".conv1")
            h = g.group_norm2d(h, blk.bn1.num_groups, _np(blk.bn1.weight), _np(blk.bn1.bias),
                               blk.bn1.eps, "relu", name=n + ".gn1")
            h = g.conv2d(h, _np(blk.conv2.weight), None, 1, blk.conv2.padding[0],
                         name=n + ".conv2")
            if blk.downsample is not None:
                dc, dn = blk.downsample[0], blk.downsample[1]
                r = g.conv2d(x, _np(dc.weight), None, dc.stride[0], 0, name=n + ".down")
                r = g.group_norm2d(r, dn.num_groups, _np(dn.weight), _np(dn.bias), dn.eps,
                                   name=n + ".down_gn")
            else:
                r = x
            x = g.group_norm2d(h, blk.bn2.num_groups, _np(blk.bn2.weight), _np(blk.bn2.bias),
                               blk.bn2.eps, "relu", res=r, name=n + ".gn2")
    pool = enc.pool
    K = pool._out_c
    if pool.nets is not None:
        sc = g.conv2d(x, _np(pool.nets.weight), _np(pool.nets.bias), 1, 0, name=name + ".kp")
    else:
        raise NotImplementedError("SpatialSoftmax without its keypoint conv")
    grid = _np(pool.pos_grid)
    kp = g.tensor(1, 2 * K)
    g.op(name + ".softmax", RG.spatial_softmax_k, x=sc.flat(),
         px=tuple(float(v) for v in grid[:, 0]), py=tuple(float(v) for v in grid[:, 1]),
         y=kp.rows())
    # (x0, y0, x1, y1, ...) is LeRobot's flatten; the kernel gives (x..., y...): permute the
    # out linear's columns instead
    wo = _np(enc.out.weight)
    perm = np.concatenate([np.arange(0, 2 * K, 2), np.arange(1, 2 * K, 2)])
    F = wo.shape[0]
    g.op(name + ".out", RG.dense_k, x=kp.rows(), w=g.weight(wo[:, perm]),
         b=g.vec(_np(enc.out.bias)), y=Tn(out_vec.base, (1, F), (F, 1)), M=1, K=2 * K,
         act="relu")
    return img
