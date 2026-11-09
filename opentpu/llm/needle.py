"""Cactus Needle (Cactus-Compute/needle, 26M "Simple Attention Network"): encoder-decoder, pure
attention (no FFN), GQA 8 / 4 heads of 64, d_model 512, vocabulary 8192 (SentencePiece), the
tied embedding. The fp32 reference forward (L1) from the checkpoint's model.safetensors (the
HF export: model.encoder.*, model.decoder.*) and the README's architecture:

    ZC(x, w)  = (1 + w) x / rms(x)                         (ZCRMSNorm, eps 1e-6)
    encoder:  x = x + sigmoid(attn_gate) SelfAttn(ZC(x))    x 12, bidirectional, RoPE 1e4;
              enc = ZC(x, final_norm)
    decoder:  x = x + sigmoid(self_attn_gate) SelfAttn(ZC(x))     causal, RoPE
              x = x + sigmoid(cross_attn_gate) CrossAttn(ZC(x), enc)   no RoPE   x 8
              logits = ZC(x, norm) E^T
    attention: q, k, v projections; q_norm / k_norm ZC per head; RoPE (rotate-half) on q, k;
               softmax(q k^T / 8) v; out_proj.

The forms the README leaves open (the encoder's input, RoPE in the cross attention, an
embedding scale) were settled against the README's own example: encoder ids = query, <tools>,
tools JSON; decoder ids = </s> (decoder_start), <tool_call>, then greedy -- the README's
'[{"name":"get_weather","arguments":{"location":"San Francisco"}}]' exactly (tests/
test_needle.py). The Needle source repository no longer has the model code (its history was
rewritten to a README); the model-folder code was not executed (needle.pkl not unpickled).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

D, NH, NK, HD, EPS, THETA = 512, 8, 4, 64, 1e-6, 10000.0
ENC_LAYERS, DEC_LAYERS = 12, 8
EOS, TOOL_CALL, TOOLS = 1, 4, 5


def load(path) -> dict:
    """model.safetensors (bf16) -> {name: fp64 array}."""
    p = Path(path)
    p = p / "model.safetensors" if p.is_dir() else p
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, e in h.items():
            if k == "__metadata__":
                continue
            if e["dtype"] != "BF16":
                raise ValueError(f"{k}: {e['dtype']}")
            a, b = e["data_offsets"]
            f.seek(base + a)
            r = np.frombuffer(f.read(b - a), np.uint16).astype(np.uint32) << 16
            out[k] = r.view(np.float32).reshape(e["shape"]).astype(np.float64)
    return out


def zc(x, w):
    return (1 + w) * x / np.sqrt(np.mean(x * x, -1, keepdims=True) + EPS)


def rope(x, pos):
    inv = 1.0 / (THETA ** (np.arange(0, HD, 2) / HD))
    a = np.outer(pos, inv)
    c, s = np.cos(a), np.sin(a)
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)


def _sig(v):
    return 1 / (1 + np.exp(-v))


def attention(W, p, xq, xkv, causal: bool, use_rope: bool):
    Tq, Tk = len(xq), len(xkv)
    q = (xq @ W[p + "q_proj.weight"].T).reshape(Tq, NH, HD).transpose(1, 0, 2)
    k = (xkv @ W[p + "k_proj.weight"].T).reshape(Tk, NK, HD).transpose(1, 0, 2)
    v = (xkv @ W[p + "v_proj.weight"].T).reshape(Tk, NK, HD).transpose(1, 0, 2)
    q, k = zc(q, W[p + "q_norm.weight"]), zc(k, W[p + "k_norm.weight"])
    if use_rope:
        q, k = rope(q, np.arange(Tq)), rope(k, np.arange(Tk))
    k, v = np.repeat(k, NH // NK, 0), np.repeat(v, NH // NK, 0)
    s = q @ k.transpose(0, 2, 1) / np.sqrt(HD)
    if causal:
        s = s + np.triu(np.full((Tq, Tk), -1e30), 1)
    s = np.exp(s - s.max(-1, keepdims=True))
    s /= s.sum(-1, keepdims=True)
    return (s @ v).transpose(1, 0, 2).reshape(Tq, NH * HD) @ W[p + "out_proj.weight"].T


def encode(W, ids) -> np.ndarray:
    x = W["model.embed_tokens.weight"][list(ids)]
    for i in range(ENC_LAYERS):
        p = f"model.encoder.layers.{i}."
        h = zc(x, W[p + "input_layernorm.weight"])
        x = x + _sig(W[p + "attn_gate"][0]) * attention(W, p + "self_attn.", h, h, False, True)
    return zc(x, W["model.encoder.final_norm.weight"])


def decode_logits(W, enc, ids) -> np.ndarray:
    """Logits [T, V] of the decoder ids (teacher forced) over the encoder output."""
    x = W["model.embed_tokens.weight"][list(ids)]
    for i in range(DEC_LAYERS):
        p = f"model.decoder.layers.{i}."
        h = zc(x, W[p + "input_layernorm.weight"])
        x = x + _sig(W[p + "self_attn_gate"][0]) * attention(W, p + "self_attn.", h, h, True, True)
        h = zc(x, W[p + "encoder_attn_layer_norm.weight"])
        x = x + _sig(W[p + "cross_attn_gate"][0]) * attention(W, p + "encoder_attn.", h, enc,
                                                              False, False)
    return zc(x, W["model.decoder.norm.weight"]) @ W["model.embed_tokens.weight"].T


def encoder_ids(tok, query: str, tools: str) -> list:
    return tok.encode(query) + [TOOLS] + tok.encode(tools)


def generate(W, enc_ids, n: int = 64) -> list:
    """Greedy decoder ids after </s> <tool_call> (up to </s>)."""
    enc = encode(W, enc_ids)
    d = [EOS, TOOL_CALL]
    for _ in range(n):
        t = int(decode_logits(W, enc, d)[-1].argmax())
        d.append(t)
        if t == EOS:
            break
    return d[2:]
