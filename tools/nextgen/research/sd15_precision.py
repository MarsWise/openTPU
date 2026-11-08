"""SD 1.5 precision study for the r8 C9 decision (FRAME level, CPU): the device's int8 MM
operands emulated in torch - W8 (per output row and 128-element K block), A8 (per row/pixel and
128-element K block: QACT's Sa), and A8 hi+lo (the activation as two int8 passes: x ~ q1 S1 +
q2 S2) - on every Linear / Conv2d of the UNet and the VAE decoder and on attention's QK^T and
PV; against the fp32 pipeline from the same seed. Reports latent / image PSNR."""
import sys, time, json
import numpy as np, torch, torch.nn.functional as Fn
from diffusers import StableDiffusionPipeline, DDIMScheduler

M, RES, STEPS, SEED = "/home/claude/models/sd15", int(sys.argv[1]), int(sys.argv[2]), 0
PROMPT = "a photograph of a red fox sitting in a snowy forest, highly detailed"
torch.set_num_threads(4)
B = 128

def qblk(x, dim, hilo=False):
    """int8 symmetric per 128-element block along `dim` (the K dimension), rne."""
    x = x.movedim(dim, -1)
    n = x.shape[-1]; pad = (-n) % B
    xp = Fn.pad(x, (0, pad)).reshape(*x.shape[:-1], -1, B)
    s = xp.abs().amax(-1, keepdim=True) / 127
    s = torch.where(s == 0, torch.ones_like(s), s)
    q = torch.round(xp / s).clamp(-127, 127) * s
    if hilo:
        r = xp - q
        s2 = r.abs().amax(-1, keepdim=True) / 127
        s2 = torch.where(s2 == 0, torch.ones_like(s2), s2)
        q = q + torch.round(r / s2).clamp(-127, 127) * s2
    return q.reshape(*x.shape[:-1], -1)[..., :n].movedim(-1, dim)

MODE = {"w": False, "a": None}          # a: None (fp32), "a8", "hilo"

def qact(x, dim):
    return x if MODE["a"] is None else qblk(x, dim, MODE["a"] == "hilo")

_lin, _conv, _sdpa = Fn.linear, Fn.conv2d, Fn.scaled_dot_product_attention
WCACHE = {}
def wq(w):
    return w                     # (W8 is applied in place once: quantize_weights)


def quantize_weights(*mods):
    """W8 in place: per output row and 128-element block of its flattened K (Cin * kh * kw)."""
    with torch.no_grad():
        for mod in mods:
            for m in mod.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    m.weight.copy_(qblk(m.weight.flatten(1), -1).view_as(m.weight))
def lin(x, w, b=None):
    return _lin(qact(x, -1), wq(w), b)
def conv(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
    # implicit GEMM: K = Cin * kh * kw; the activation quantized per pixel over channels
    return _conv(qact(x, 1), wq(w), b, stride, padding, dilation, groups)
def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    s = (scale or q.shape[-1] ** -0.5)
    z = qact(q, -1) @ qact(k, -1).transpose(-1, -2) * s
    if is_causal:
        L, S = z.shape[-2:]
        z = z.masked_fill(torch.ones(L, S, dtype=torch.bool).triu(1), float("-inf"))
    if attn_mask is not None:
        z = z.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else z + attn_mask
    a = torch.softmax(z, -1)
    return qact(a, -1) @ qact(v, -2)
Fn.linear, Fn.conv2d, Fn.scaled_dot_product_attention = lin, conv, sdpa
torch.nn.functional.linear = lin; torch.nn.functional.conv2d = conv
torch.nn.functional.scaled_dot_product_attention = sdpa
import torch.nn as nn
nn.Linear.forward = lambda self, x: lin(x, self.weight, self.bias)
nn.Conv2d.forward = lambda self, x: (conv(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                     if self.padding_mode == "zeros" else self._conv_forward(x, self.weight, self.bias))

pipe = StableDiffusionPipeline.from_pretrained(M, torch_dtype=torch.float32, safety_checker=None, feature_extractor=None,
                                               requires_safety_checker=False, variant="fp16")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)
out, res = {}, {}
for name, w, a in (("fp32", False, None), ("w8a8", True, "a8"), ("w8a16_hilo", True, "hilo")):
    if w and not MODE["w"]:
        quantize_weights(pipe.unet, pipe.vae.decoder, pipe.vae.post_quant_conv)
    MODE["w"], MODE["a"] = w, a
    t = time.time()
    g = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        r = pipe(PROMPT, height=RES, width=RES, num_inference_steps=STEPS, guidance_scale=7.5,
                 generator=g, output_type="np")
    img = r.images[0]
    out[name] = img
    import PIL.Image
    PIL.Image.fromarray((img * 255).round().astype(np.uint8)).save(
        f"/tmp/claude-0/-home-claude/d587f972-39a3-5f63-8562-297a7259d9a3/scratchpad/sd15_{name}_{RES}.png")
    if name != "fp32":
        mse = float(((img - out["fp32"]) ** 2).mean())
        res[name] = {"psnr_db": 10 * np.log10(1 / mse) if mse else 99.0,
                     "max_abs": float(np.abs(img - out["fp32"]).max())}
    print(name, f"{time.time() - t:.0f}s", res.get(name), flush=True)
print(json.dumps({"res": RES, "steps": STEPS, "seed": SEED, "prompt": PROMPT, **res}))
