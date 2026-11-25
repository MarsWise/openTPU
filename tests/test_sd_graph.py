"""Stable Diffusion 1.5's device graph (opentpu/sd_graph.py): the schedule and time embedding
against diffusers, the guidance + DDIM step, GEGLU and the arena's zero fill on the ISA; a small
random UNet / VAE pair against diffusers' modules on the same weights (one ResNet + transformer
level, the real blocks' code paths); and, when SD 1.5's weights are here, CLIP against
transformers."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from opentpu import resident as RG
from opentpu import sd_graph as S
from opentpu.isasim import lean_config

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
SD_DIR = Path(os.environ.get("OTPU_SD15_DIR", "/mnt/user-data/uploads/ai-models/diffusion/sd-v1-5"))


def _cfg(dram=1 << 26):
    return lean_config(DRAM_BYTES=dram)          # the card's (r5x lean) configuration


def _cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def test_schedule_is_diffusers_ddim():
    """DDIMScheduler (SD 1.5's config: scaled_linear, steps_offset 1, set_alpha_to_one False):
    the timesteps, and x_prev = a x + b eps is its step (eta 0) for any x, eps."""
    from diffusers import DDIMScheduler
    sc = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                       steps_offset=1, set_alpha_to_one=False, clip_sample=False)
    for steps in (2, 20):
        sc.set_timesteps(steps)
        ours = S.ddim_schedule(steps)
        assert [t for t, _, _ in ours] == sc.timesteps.tolist()
        rng = np.random.default_rng(steps)
        for t, a, b in ours[:3]:
            x, e = (torch.as_tensor(rng.normal(size=(1, 4, 2, 2)), dtype=torch.float64)
                    for _ in range(2))
            want = sc.step(e, t, x).prev_sample.numpy()
            np.testing.assert_allclose(a * x.numpy() + b * e.numpy(), want, rtol=1e-5, atol=1e-6)


def test_time_embedding_row_is_diffusers():
    from diffusers.models.embeddings import get_timestep_embedding
    for t in (1, 501, 951):
        want = get_timestep_embedding(torch.tensor([t]), 320, flip_sin_to_cos=True,
                                      downscale_freq_shift=0)[0].numpy()
        np.testing.assert_allclose(S.timestep_row(t), want, atol=1e-4)   # (fp32 exp ulps x t)


def test_guidance_ddim_geglu_and_zero_fill_on_the_isa():
    rng = np.random.default_rng(1)
    g = S.SDGraph(_cfg(), 1 << 25)
    x, eu, ec, y = (g.tensor(16, 4) for _ in range(4))
    g.op("ddim", S.cfg_ddim_k, x=x.rows(), eu=eu.rows(), ec=ec.rows(), y=y.rows(), gs=7.5,
         a=0.97, b=-0.21)
    h, gg, z = g.tensor(8, 64), g.tensor(8, 64), g.tensor(8, 64)
    g.op("geglu", S.geglu_k, h=h.rows(), g=gg.rows(), y=z.rows())
    junk = g.tensor(300, 50)
    S.zero_region(g, junk.addr, junk.addr + junk.nbytes, len(g.ops), "zero")
    img = g.image()
    vals = {t: rng.normal(size=(t.L, t.C)).astype(np.float32) for t in (x, eu, ec, h, gg, junk)}
    for t, v in vals.items():
        g.put(img, t, v)
    img, _ = RG.run_isa(g.cfg, g.compile(), img)
    want = 0.97 * vals[x] - 0.21 * (vals[eu] + 7.5 * (vals[ec] - vals[eu]))
    np.testing.assert_allclose(g.read(img, y), want, rtol=1e-5, atol=1e-5)
    gl = torch.nn.functional.gelu(torch.as_tensor(vals[gg])).numpy()
    np.testing.assert_allclose(g.read(img, z), vals[h] * gl, atol=2e-3)
    assert not g.read(img, junk).any()


def _tiny_unet():
    """A small UNet2DConditionModel with SD 1.5's block types (two levels: cross-attention
    down / up blocks, a plain one, the mid block), 32-channel GroupNorm, 8 heads."""
    from diffusers import UNet2DConditionModel
    torch.manual_seed(0)
    m = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4, layers_per_block=2,
                             block_out_channels=(64, 128),
                             down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                             up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
                             cross_attention_dim=64, attention_head_dim=8, norm_num_groups=32)
    return m.eval()


def test_small_unet_follows_diffusers():
    """The device UNet (SD.unet's code: resnets with the time embedding, transformers with
    self / cross attention and GEGLU, down / up sampling, skip concatenations) on a small
    random diffusers UNet of SD 1.5's block types, latent 8x8: eps cosine > 0.995."""
    m = _tiny_unet()
    W = {"unet." + k: v.detach().numpy() for k, v in m.state_dict().items()}
    rng = np.random.default_rng(2)
    x0 = rng.normal(size=(64, 4)).astype(np.float32)
    ctx0 = rng.normal(size=(9, 64)).astype(np.float32)
    g = S.SDGraph(_cfg(1 << 28), 1 << 27)
    sd = S.SD(W)
    x, ctx, tr = g.tensor(64, 4), g.tensor(9, 64), g.tensor(1, 64)
    # (the tiny UNet's time embedding is 64 wide: block_out_channels[0])
    eps = _unet_levels(sd, g, S._t2(x, 8, 8), tr, ctx, levels=2)
    img = g.image()
    t = 401
    from diffusers.models.embeddings import get_timestep_embedding
    trow = get_timestep_embedding(torch.tensor([t]), 64, flip_sin_to_cos=True,
                                  downscale_freq_shift=0)[0].numpy()
    g.put(img, x, x0)
    g.put(img, ctx, ctx0)
    g.put(img, tr, trow[None, :])
    img, _ = RG.run_isa(g.cfg, g.compile(), img)
    dev = g.read(img, S._dt(eps))
    with torch.no_grad():
        ref = m(torch.as_tensor(x0).reshape(1, 8, 8, 4).permute(0, 3, 1, 2), torch.tensor([t]),
                encoder_hidden_states=torch.as_tensor(ctx0)[None]).sample
    ref = ref[0].permute(1, 2, 0).reshape(64, 4).numpy()
    assert _cos(dev, ref) > 0.995


def test_reused_arena_needs_no_zeroing_between_equal_layouts():
    """build_txt2img zeroes the arena only before an evaluation whose layout differs from the
    previous one's: a second UNet evaluation on the first one's arena (its activations left
    there) gives the output a fresh arena gives, bit for bit, and the two layouts are equal."""
    m = _tiny_unet()
    W = {"unet." + k: v.detach().numpy() for k, v in m.state_dict().items()}
    rng = np.random.default_rng(5)
    xa, xb = (rng.normal(size=(64, 4)).astype(np.float32) for _ in range(2))
    ctx0 = rng.normal(size=(9, 64)).astype(np.float32)
    trow = rng.normal(size=(1, 64)).astype(np.float32)

    def graph(inputs):
        g = S.SDGraph(_cfg(1 << 28), 1 << 27)
        sd = S.SD(W)
        ctx, tr = g.tensor(9, 64), g.tensor(1, 64)
        xs = [g.tensor(64, 4) for _ in inputs]
        outs = [g.tensor(64, 4) for _ in inputs]
        arena = -(-g.top // RG.ALIGN) * RG.ALIGN
        sigs = []
        for x, o in zip(xs, outs):
            g.top = arena
            at = len(g.ops)
            e = _unet_levels(sd, g, S._t2(x, 8, 8), tr, ctx, levels=2)
            g.copy(S._dt(e), o, name="keep")
            sigs.append(S._layout(g.ops[at:], arena))
        img = g.image()
        for x, v in zip(xs, inputs):
            g.put(img, x, v)
        g.put(img, ctx, ctx0)
        g.put(img, tr, trow)
        img, _ = RG.run_isa(g.cfg, g.compile(), img)
        return [g.read(img, o) for o in outs], sigs

    (a1, b2), sigs = graph([xa, xb])
    (b1,), _ = graph([xb])
    assert sigs[0] == sigs[1]
    assert np.array_equal(b2, b1)
    assert not np.array_equal(a1, b1)


def _unet_levels(sd, g, x, t_in, ctx, levels):
    """SD.unet for a UNet of `levels` down / up levels (the real one has 4)."""
    import types
    real = S.SD.unet

    def unet(self, g, x, t_in, ctx, tag="u"):
        U, eps = "unet.", 1e-5
        temb = self.time_embedding(g, t_in)
        h = self.conv(g, x, U + "conv_in.", name=f"{tag}.conv_in")
        skips = [h]
        for i in range(levels):
            D = f"{U}down_blocks.{i}."
            for j in range(2):
                h = self.resnet(g, h, temb, f"{D}resnets.{j}.", eps, f"{tag}.d{i}r{j}")
                if self.has(f"{D}attentions.{j}.norm.weight"):
                    h = self.transformer(g, h, ctx, f"{D}attentions.{j}.", f"{tag}.d{i}a{j}")
                skips.append(h)
            if i < levels - 1:
                h = self.conv(g, h, f"{D}downsamplers.0.conv.", stride=2, name=f"{tag}.d{i}down")
                skips.append(h)
        M = U + "mid_block."
        h = self.resnet(g, h, temb, M + "resnets.0.", eps, f"{tag}.m.r0")
        h = self.transformer(g, h, ctx, M + "attentions.0.", f"{tag}.m.a0")
        h = self.resnet(g, h, temb, M + "resnets.1.", eps, f"{tag}.m.r1")
        for i in range(levels):
            Up = f"{U}up_blocks.{i}."
            for j in range(3):
                h = self.concat(g, h, skips.pop(), f"{tag}.u{i}c{j}")
                h = self.resnet(g, h, temb, f"{Up}resnets.{j}.", eps, f"{tag}.u{i}r{j}")
                if self.has(f"{Up}attentions.{j}.norm.weight"):
                    h = self.transformer(g, h, ctx, f"{Up}attentions.{j}.", f"{tag}.u{i}a{j}")
            if i < levels - 1:
                h = self.conv(g, self.upsample2(g, h, f"{tag}.u{i}up"),
                              f"{Up}upsamplers.0.conv.", name=f"{tag}.u{i}upconv")
        h = self.gn(g, h, U + "conv_norm_out.", eps, "silu", name=f"{tag}.gn_out")
        return self.conv(g, h, U + "conv_out.", name=f"{tag}.conv_out")
    assert real is S.SD.unet
    return types.MethodType(unet, sd)(g, x, t_in, ctx)


def test_clip_follows_transformers():
    """SD 1.5's CLIP text encoder (causal attention, quick GELU) on its fp16 weights: every
    token row's cosine > 0.995 (measured min 0.9981, mean 0.9994)."""
    te = SD_DIR / "text_encoder/model.fp16.safetensors"
    if not te.exists():
        pytest.skip(f"no SD 1.5 text encoder in {SD_DIR}")
    from safetensors.torch import load_file
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
    tok = CLIPTokenizer(str(SD_DIR / "tokenizer/vocab.json"), str(SD_DIR / "tokenizer/merges.txt"))
    ids = tok("a photo of a red fox in the snow", padding="max_length", max_length=77).input_ids
    g = S.SDGraph(_cfg(1 << 29), 1 << 27)
    out = S.SD(S.Weights(text=te)).clip(g, ids)
    img, _ = RG.run_isa(g.cfg, g.compile(), g.image())
    dev = g.read(img, out)
    c = CLIPTextConfig(**{k: v for k, v in json.loads((SD_DIR / "text_encoder/config.json")
                                                      .read_text()).items() if not k.startswith("_")})
    m = CLIPTextModel(c).eval()
    m.load_state_dict({k.replace("text_model.", ""): v.float() for k, v in load_file(str(te)).items()},
                      strict=False)
    with torch.no_grad():
        ref = m(torch.tensor([ids])).last_hidden_state[0].numpy()
    cos = (dev * ref).sum(1) / np.linalg.norm(dev, axis=1) / np.linalg.norm(ref, axis=1)
    assert cos.min() > 0.995


# ---------------------------------------------------------------- LCM-LoRA and TAESD
def test_lcm_schedule_is_diffusers_lcm():
    """LCMScheduler from SD 1.5's config: the timesteps, and x_prev = a x + b eps + c z is its
    step for any x, eps with z the noise it draws (none on the last step)."""
    from diffusers import LCMScheduler
    sc = LCMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                      steps_offset=1, set_alpha_to_one=False, clip_sample=False)
    for steps in (1, 2, 4, 8):
        sc.set_timesteps(steps)
        ours = S.lcm_schedule(steps)
        assert [t for t, _, _, _ in ours] == sc.timesteps.tolist()
        assert ours[-1][3] == 0.0
        rng = np.random.default_rng(steps)
        for i, (t, a, b, c) in enumerate(ours):
            x, e = (torch.as_tensor(rng.normal(size=(1, 4, 2, 2)), dtype=torch.float64)
                    for _ in range(2))
            want = sc.step(e, t, x, generator=torch.Generator().manual_seed(i)).prev_sample
            z = torch.randn((1, 4, 2, 2), generator=torch.Generator().manual_seed(i),
                            dtype=torch.float64)
            got = a * x.numpy() + b * e.numpy() + c * z.numpy()
            # (diffusers' alphas_cumprod is an fp32 table)
            np.testing.assert_allclose(got, want.numpy(), rtol=2e-5, atol=2e-6)


def test_lcm_step_guidance_and_softclamp_on_the_isa():
    rng = np.random.default_rng(3)
    g = S.SDGraph(_cfg(), 1 << 25)
    x, e, z, y, y2, eu, ec, gy, c, cy = (g.tensor(16, 4) for _ in range(10))
    g.op("lcm", S.lcm_k, x=x.rows(), e=e.rows(), z=z.rows(), y=y.rows(), a=1.3, b=-0.4, c=0.7)
    g.op("lcm.last", S.lcm_k, x=x.rows(), e=e.rows(), y=y2.rows(), a=0.9, b=-0.2)
    g.op("cfg", S.cfg_k, eu=eu.rows(), ec=ec.rows(), y=gy.rows(), gs=1.5)
    g.op("clamp", S.softclamp_k, x=c.rows(), y=cy.rows(), c=3.0)
    img = g.image()
    v = {t: (rng.normal(size=(16, 4)) * 4).astype(np.float32) for t in (x, e, z, eu, ec, c)}
    for t, a in v.items():
        g.put(img, t, a)
    img, _ = RG.run_isa(g.cfg, g.compile(), img)
    np.testing.assert_allclose(g.read(img, y), 1.3 * v[x] - 0.4 * v[e] + 0.7 * v[z],
                               rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(g.read(img, y2), 0.9 * v[x] - 0.2 * v[e], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(g.read(img, gy), v[eu] + 1.5 * (v[ec] - v[eu]), rtol=1e-5,
                               atol=1e-5)
    np.testing.assert_allclose(g.read(img, cy), 3 * np.tanh(v[c] / 3), atol=2e-3)


def _taesd_weights():
    """AutoencoderTiny's decoder: madebyollin/taesd's when OTPU_TAESD names its safetensors,
    else diffusers' random initialization of the same config."""
    from diffusers import AutoencoderTiny
    torch.manual_seed(0)
    m = AutoencoderTiny().eval()                # (its defaults: TAESD's config)
    p = os.environ.get("OTPU_TAESD")
    if p:
        from safetensors.torch import load_file
        m.load_state_dict(load_file(p))
    return m, {"taesd." + k: v.detach().numpy() for k, v in m.state_dict().items()
               if k.startswith("decoder.")}


def test_taesd_decoder_follows_diffusers():
    """SD.taesd_decode (clamp, conv + ReLU, residual blocks with the ReLU after the residual
    in the convolution's program, nearest x2, the x 2 - 1 folded into the last conv) against
    AutoencoderTiny.decode, latent 4x4 -> 32x32."""
    m, W = _taesd_weights()
    rng = np.random.default_rng(4)
    z0 = (rng.normal(size=(16, 4)) * 2).astype(np.float32)
    g = S.SDGraph(_cfg(1 << 27), 1 << 26)
    z = g.tensor(16, 4)
    out = S.SD(W).taesd_decode(g, S._t2(z, 4, 4))
    img = g.image()
    g.put(img, z, z0)
    img, _ = RG.run_isa(g.cfg, g.compile(), img)
    dev = g.read(img, S._dt(out)).reshape(32, 32, 3)
    with torch.no_grad():
        ref = m.decode(torch.as_tensor(z0).reshape(1, 4, 4, 4).permute(0, 3, 1, 2)).sample
    ref = ref[0].permute(1, 2, 0).numpy()
    assert _cos(dev, ref) > 0.995, _cos(dev, ref)
    mse = float(((np.clip(dev, -1, 1) - np.clip(ref, -1, 1)) ** 2).mean())
    assert 10 * np.log10(4 / mse) > 30          # (TAESD's weights: 36 dB, int8 MXU)


def test_lora_merge_is_kohya_and_names_diffusers_modules():
    """A kohya LoRA (LCM-LoRA's format: lora_unet_<path>, alpha, rank) merged as read: W +
    alpha / rank up @ down for linear, 1x1 and 3x3 weights; the module each changes is the
    one diffusers' own kohya conversion names; an unknown module is refused."""
    import tempfile
    from safetensors.numpy import save_file
    from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_lora_to_diffusers
    rng = np.random.default_rng(6)
    base = {"down_blocks.0.attentions.0.proj_in.weight": (32, 32, 1, 1),
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight": (32, 32),
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0.weight": (32, 32),
            "down_blocks.0.resnets.0.conv1.weight": (32, 16, 3, 3),
            "down_blocks.0.resnets.0.conv1.bias": (32,)}
    U = {k: rng.normal(size=s).astype(np.float32) for k, s in base.items()}
    L, r = {}, 4
    for k, s in base.items():
        if not k.endswith(".weight"):
            continue
        n = "lora_unet_" + k[:-7].replace(".", "_")
        kin = s[1] * (s[2] * s[3] if len(s) == 4 else 1)
        L[n + ".lora_down.weight"] = rng.normal(size=(r, s[1]) + s[2:]).astype(np.float32)
        L[n + ".lora_up.weight"] = rng.normal(size=(s[0], r) + ((1, 1) if len(s) == 4 else ())
                                              ).astype(np.float32)
        L[n + ".alpha"] = np.array(2.0, np.float32)
        assert L[n + ".lora_down.weight"].reshape(r, -1).shape[1] == kin
    with tempfile.TemporaryDirectory() as d:
        save_file(U, f"{d}/unet.safetensors")
        save_file(L, f"{d}/lora.safetensors")
        W = S.Weights(unet=[f"{d}/unet.safetensors"], lora=f"{d}/lora.safetensors",
                      lora_scale=0.5)
        for k in base:
            got = W["unet." + k]
            n = "lora_unet_" + k[:-7].replace(".", "_")
            if k.endswith(".bias"):
                np.testing.assert_array_equal(got, U[k])
                continue
            up, dn = L[n + ".lora_up.weight"], L[n + ".lora_down.weight"]
            want = U[k] + 0.5 * 2.0 / r * (up.reshape(up.shape[0], r) @ dn.reshape(r, -1)
                                           ).reshape(U[k].shape)
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
        conv, _ = _convert_non_diffusers_lora_to_diffusers(
            {k: torch.as_tensor(v) for k, v in L.items()})
        mods = {k.replace(".processor.", ".").replace("_lora.", ".lora.").split(".lora")[0]
                .removeprefix("unet.") for k in conv if "lora" in k}
        # (diffusers' attention processor calls to_out.0's LoRA to_out_lora)
        assert mods == {k[5:-7].replace(".to_out.0", ".to_out") for k in W.lora}
        dn0 = next(v for k, v in L.items() if k.endswith(".lora_down.weight"))
        save_file({**L, "lora_unet_no_such_module.lora_down.weight": dn0},
                  f"{d}/bad.safetensors")
        with pytest.raises(ValueError, match="name no UNet weight"):
            S.Weights(unet=[f"{d}/unet.safetensors"], lora=f"{d}/bad.safetensors")


def test_lcm_taesd_pipeline_on_the_isa():
    """build_txt2img(sampler="lcm", decoder="taesd") with stand-ins for CLIP and the UNet
    (eps = 0.3 x): one evaluation a step (guidance 1), the initial latent and each step's noise
    drawn on the device (keys 1, 2 + step, none on the last step), LCMScheduler's coefficients,
    then TAESD -- against the same chain in NumPy and AutoencoderTiny.decode."""
    from opentpu.kernels import randn as RN
    m, W = _taesd_weights()

    class Stub(S.SD):
        calls = []

        def clip(self, g, ids):
            self.calls.append("clip")
            return g.tensor(77, 8)

        def unet(self, g, x, t_in, ctx, tag="u"):
            self.calls.append(tag)
            return S._t2(g.scale(S._dt(x), 0.3, y=g.tensor(x.H * x.W, 4), name=tag), x.H, x.W)

    h = w = 4
    steps, seed = 4, 42
    g = S.SDGraph(_cfg(1 << 27), 1 << 26)
    sd = Stub(W)
    out, seed_t, _ = S.build_txt2img(g, sd, [0] * 77, [0] * 77, h, w, steps, 1.0,
                                     sampler="lcm", decoder="taesd")
    assert sd.calls == ["clip"] + [f"s{s}c" for s in range(steps)]
    assert sum(1 for n, _, _ in g.ops if n.startswith("sd.noise")) == steps   # 1 + 3 steps
    img = g.image()
    RG.Graph.put(None, img, seed_t, np.array([seed], np.uint32).view(np.float32))
    img, _ = RG.run_isa(g.cfg, g.compile(), img)
    dev = g.read(img, S._dt(out)).reshape(8 * h, 8 * w, 3)
    x = RN.seeded_reference(1, h * w * 4, seed, 1).reshape(h * w, 4).astype(np.float64)
    for s, (t, a, b, c) in enumerate(S.lcm_schedule(steps)):
        x = a * x + b * 0.3 * x
        if c:
            x = x + c * RN.seeded_reference(1, h * w * 4, seed, 2 + s).reshape(h * w, 4)
    with torch.no_grad():
        ref = m.decode(torch.as_tensor(x, dtype=torch.float32).reshape(1, h, w, 4)
                       .permute(0, 3, 1, 2)).sample[0].permute(1, 2, 0).numpy()
    assert _cos(dev, ref) > 0.995, _cos(dev, ref)
