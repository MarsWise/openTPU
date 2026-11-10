"""PaddleOCR-VL 1.6 (PaddlePaddle, 0.9B: a NaViT SigLIP vision tower, a 2 x 2 merge projector,
the ERNIE-4.5 0.3B decoder): the fp32 reference forward (L1), written from the model folder's
modeling_paddleocr_vl.py and image_processing_paddleocr_vl.py (read, not executed):

  image:   smart_resize to multiples of 28 within [min, max] pixels, bicubic, x / 255,
           (x - 0.5) / 0.5, 14 x 14 patches in raster order (h, w)
  tower:   patch conv (a linear of the flattened patch) + bias + the 27 x 27 position table
           bilinearly resized to (h, w) (align_corners False); 27 pre-norm layers: LayerNorm,
           attention (bias, 16 heads of 72, bidirectional, 2-D RoPE: per head the first 36
           dims rotate-half rotated by (row freqs, col freqs) of theta 1e4 over 18 frequencies
           each, repeated twice), LayerNorm, fc1 gelu_tanh fc2; post_layernorm
  projector (mlp_AR): LayerNorm per patch row, each 2 x 2 window's 4 rows as one [4608] row,
           linear_1, GELU (erf), linear_2 -> 1024
  decoder: ERNIE-4.5: 18 layers RMSNorm (1e-5), GQA 16 / 2 heads of 128 (no bias, no q / k
           norms), mRoPE (theta 5e5, sections 16 / 24 / 24 of the 64 frequencies: t, h, w),
           SiLU MLP; final norm; lm_head (untied)
  prompt:  <|begin_of_sentence|>User: <|IMAGE_START|> image tokens <|IMAGE_END|>{task}\\n
           Assistant:\\n  (task "OCR:"); the image tokens' positions (t 0, h, w) + offset, the
           text after them from the image's max + 1
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np

PATCH, MERGE, MIN_PIX, MAX_PIX = 14, 2, 112896, 1003520
IMAGE_TOKEN, VISION_START, VISION_END = 100295, 101305, 101306
MROPE = (16, 24, 24)


class Weights:
    """model.safetensors (bf16) as torch fp32 tensors, read on demand."""

    def __init__(self, path):
        p = Path(path)
        self.path = p / "model.safetensors" if p.is_dir() else p
        with open(self.path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.h = json.loads(f.read(n))
        self.base = 8 + n

    def __getitem__(self, k):
        import torch
        e = self.h[k]
        a, b = e["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.base + a)
            r = np.frombuffer(f.read(b - a), np.uint16).astype(np.uint32) << 16
        return torch.from_numpy(r.view(np.float32).reshape(e["shape"]).copy())

    def keys(self):
        return [k for k in self.h if k != "__metadata__"]


def smart_resize(h: int, w: int, factor: int = 28, min_pixels: int = MIN_PIX,
                 max_pixels: int = MAX_PIX):
    if h < factor:
        w, h = round(w * factor / h), factor
    if w < factor:
        h, w = round(h * factor / w), factor
    hb, wb = round(h / factor) * factor, round(w / factor) * factor
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb, wb = math.floor(h / beta / factor) * factor, math.floor(w / beta / factor) * factor
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb, wb = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return hb, wb


def preprocess(img):
    """PIL image -> (patches [N, 3 * 14 * 14] fp32 in raster order, (h, w) in patches)."""
    from PIL import Image
    img = img.convert("RGB")
    H, W = smart_resize(img.height, img.width)
    a = np.asarray(img.resize((W, H), Image.BICUBIC), np.float32)
    a = (a / 255.0 - 0.5) / 0.5
    gh, gw = H // PATCH, W // PATCH
    p = a.reshape(gh, PATCH, gw, PATCH, 3).transpose(0, 2, 4, 1, 3).reshape(gh * gw, -1)
    return np.ascontiguousarray(p, np.float32), (gh, gw)


def prompt_ids(tok, gh: int, gw: int, task: str = "OCR:") -> list:
    """The chat template's ids with the image's (gh / 2) (gw / 2) placeholders."""
    n = (gh // MERGE) * (gw // MERGE)
    head = tok.encode("<|begin_of_sentence|>User: ", add_special_tokens=False).ids
    tail = tok.encode(f"{task}\nAssistant:\n", add_special_tokens=False).ids
    return head + [VISION_START] + [IMAGE_TOKEN] * n + [VISION_END] + tail


def mrope_positions(ids, gh: int, gw: int) -> np.ndarray:
    """[3, T] positions (get_rope_index for one image, t = 1)."""
    lh, lw = gh // MERGE, gw // MERGE
    s = ids.index(IMAGE_TOKEN)
    n = lh * lw
    pos = np.zeros((3, len(ids)), np.int64)
    pos[:, :s] = np.arange(s)
    pos[0, s:s + n] = s
    pos[1, s:s + n] = s + np.repeat(np.arange(lh), lw)
    pos[2, s:s + n] = s + np.tile(np.arange(lw), lh)
    m = pos[:, :s + n].max() + 1
    pos[:, s + n:] = m + np.arange(len(ids) - s - n)
    return pos


def mrope_tables(pos3, d: int = 128, theta: float = 500000.0):
    """cos / sin [T, d] (rotate-half layout) of the 3-D positions."""
    inv = 1.0 / (theta ** (np.arange(0, d, 2, dtype=np.float64) / d))   # 64
    f = pos3[:, :, None].astype(np.float64) * inv                       # [3, T, 64]
    emb = np.concatenate([f, f], -1)                                    # [3, T, 128]
    sec = list(MROPE) * 2
    parts, o = [], 0
    for i, n in enumerate(sec):
        parts.append(emb[i % 3, :, o:o + n])
        o += n
    e = np.concatenate(parts, -1)
    return np.cos(e).astype(np.float32), np.sin(e).astype(np.float32)


def vision_rope(gh: int, gw: int, hd: int = 72, theta: float = 10000.0):
    """cos / sin [N, hd] of the tower's 2-D RoPE (raster order)."""
    dim = hd // 2
    inv = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    hh = np.repeat(np.arange(gh), gw)
    ww = np.tile(np.arange(gw), gh)
    e = np.concatenate([np.outer(hh, inv), np.outer(ww, inv)], -1)      # [N, 36]
    e = np.concatenate([e, e], -1)                                       # [N, 72]
    return np.cos(e).astype(np.float32), np.sin(e).astype(np.float32)


def _rot_half(x):
    import torch
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], -1)


def vision_forward(W, patches, gh: int, gw: int):
    """The tower and the projector: [N / 4, 1024] image rows (merge-window order: raster over
    the (gh / 2, gw / 2) grid)."""
    import torch
    import torch.nn.functional as Fn
    v = "visual.vision_model."
    x = torch.from_numpy(patches)
    pw = W[v + "embeddings.patch_embedding.weight"].reshape(1152, -1)
    x = x @ pw.T + W[v + "embeddings.patch_embedding.bias"]
    pe = W[v + "embeddings.position_embedding.weight"].reshape(1, 27, 27, 1152).permute(0, 3, 1, 2)
    pe = Fn.interpolate(pe, size=(gh, gw), mode="bilinear", align_corners=False)
    x = x + pe.permute(0, 2, 3, 1).reshape(-1, 1152)
    cos, sin = (torch.from_numpy(t)[:, None, :] for t in vision_rope(gh, gw))
    N, H, nh, hd = x.shape[0], 1152, 16, 72
    for i in range(27):
        p = f"{v}encoder.layers.{i}."
        h = Fn.layer_norm(x, (H,), W[p + "layer_norm1.weight"], W[p + "layer_norm1.bias"], 1e-6)
        q = (h @ W[p + "self_attn.q_proj.weight"].T + W[p + "self_attn.q_proj.bias"]).view(N, nh, hd)
        k = (h @ W[p + "self_attn.k_proj.weight"].T + W[p + "self_attn.k_proj.bias"]).view(N, nh, hd)
        vv = (h @ W[p + "self_attn.v_proj.weight"].T + W[p + "self_attn.v_proj.bias"]).view(N, nh, hd)
        q, k = q * cos + _rot_half(q) * sin, k * cos + _rot_half(k) * sin
        a = torch.softmax(torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(hd), -1)
        o = torch.einsum("hqk,khd->qhd", a, vv).reshape(N, H)
        x = x + o @ W[p + "self_attn.out_proj.weight"].T + W[p + "self_attn.out_proj.bias"]
        h = Fn.layer_norm(x, (H,), W[p + "layer_norm2.weight"], W[p + "layer_norm2.bias"], 1e-6)
        h = Fn.gelu(h @ W[p + "mlp.fc1.weight"].T + W[p + "mlp.fc1.bias"], approximate="tanh")
        x = x + h @ W[p + "mlp.fc2.weight"].T + W[p + "mlp.fc2.bias"]
    x = Fn.layer_norm(x, (H,), W[v + "post_layernorm.weight"], W[v + "post_layernorm.bias"], 1e-6)
    x = Fn.layer_norm(x, (H,), W["mlp_AR.pre_norm.weight"], W["mlp_AR.pre_norm.bias"], 1e-5)
    x = x.view(gh // 2, 2, gw // 2, 2, H).permute(0, 2, 1, 3, 4).reshape(-1, 4 * H)
    x = Fn.gelu(x @ W["mlp_AR.linear_1.weight"].T + W["mlp_AR.linear_1.bias"])
    return x @ W["mlp_AR.linear_2.weight"].T + W["mlp_AR.linear_2.bias"]


class Decoder:
    """ERNIE-4.5 0.3B over embedding rows with mRoPE cos / sin rows, a growing KV (fp32)."""

    def __init__(self, W):
        self.W = W
        self.L = [{k: W[f"model.layers.{i}.{k}"] for k in (
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
            "mlp.down_proj.weight")} for i in range(18)]
        self.norm, self.head = W["model.norm.weight"], W["lm_head.weight"]
        self.embed = W["model.embed_tokens.weight"]
        self.k, self.v = [None] * 18, [None] * 18

    @staticmethod
    def _rms(x, w):
        import torch
        return w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)

    def forward(self, x, cos, sin):
        """x [T, 1024], cos / sin [T, 128] -> logits [T, V] (appending to the KV)."""
        import torch
        T = x.shape[0]
        c, s = torch.from_numpy(cos)[:, None], torch.from_numpy(sin)[:, None]
        for i, w in enumerate(self.L):
            h = self._rms(x, w["input_layernorm.weight"])
            q = (h @ w["self_attn.q_proj.weight"].T).view(T, 16, 128)
            k = (h @ w["self_attn.k_proj.weight"].T).view(T, 2, 128)
            v = (h @ w["self_attn.v_proj.weight"].T).view(T, 2, 128)
            q, k = q * c + _rot_half(q) * s, k * c + _rot_half(k) * s
            self.k[i] = k if self.k[i] is None else torch.cat([self.k[i], k])
            self.v[i] = v if self.v[i] is None else torch.cat([self.v[i], v])
            K, V = (t.repeat_interleave(8, 1) for t in (self.k[i], self.v[i]))
            S = K.shape[0]
            sc = torch.einsum("qhd,khd->hqk", q, K) / math.sqrt(128)
            mask = torch.triu(torch.full((T, S), float("-inf")), S - T + 1)
            a = torch.softmax(sc + mask, -1)
            o = torch.einsum("hqk,khd->qhd", a, V).reshape(T, -1)
            x = x + o @ w["self_attn.o_proj.weight"].T
            h = self._rms(x, w["post_attention_layernorm.weight"])
            g = h @ w["mlp.gate_proj.weight"].T
            x = x + (torch.nn.functional.silu(g) * (h @ w["mlp.up_proj.weight"].T)) @ \
                w["mlp.down_proj.weight"].T
        return self._rms(x, self.norm) @ self.head.T


def ocr(W, tok, img, n: int = 64, task: str = "OCR:"):
    """Greedy decode of the image (the reference) -> (ids, text)."""
    import torch
    with torch.no_grad():
        patches, (gh, gw) = preprocess(img)
        rows = vision_forward(W, patches, gh, gw)
        ids = prompt_ids(tok, gh, gw, task)
        pos = mrope_positions(ids, gh, gw)
        x = Decoder(W).embed[ids].clone()
        x[torch.tensor(ids) == IMAGE_TOKEN] = rows
        dec = Decoder(W)
        cos, sin = mrope_tables(pos)
        lg = dec.forward(x, cos, sin)[-1]
        out, p = [], int(pos.max()) + 1
        for _ in range(n):
            t = int(lg.argmax())
            out.append(t)
            if t == 2:                              # </s>
                break
            c, s = mrope_tables(np.full((3, 1), p))
            lg = dec.forward(dec.embed[[t]], c, s)[-1]
            p += 1
        return out, tok.decode([t for t in out if t != 2])
