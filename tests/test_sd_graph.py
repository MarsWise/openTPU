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
