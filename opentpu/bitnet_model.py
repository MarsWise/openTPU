"""Hybrid BitNet b1.58 inference: the ternary projections run as TMM (ISA simulator or the board),
RMSNorm / RoPE / attention / relu^2 / the tied LM head run on the host in numpy.

The block structure follows HF `modeling_bitnet.py`:
    h += o_proj(attn_sub_norm(attn(rope(q), rope(k), v)))   after input_layernorm
    h += down_proj(ffn_sub_norm(relu(gate(x))^2 * up(x)))   after post_attention_layernorm
Projection weights are the checkpoint's packed uint8 [N/4, K] tensors with a scalar weight_scale.
Backends differ only in how y = quant8(x) @ W^T * scale is computed; the board and ISA backends
use QACT's per-D-block power-of-two activation scale, the reference backend HF's per-token absmax.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import numpy as np

from opentpu import isa as I
from opentpu.bitnet import hf_to_tmm, reference_linear, unpack_hf_bitnet

PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass
class BitNetConfig:
    hidden: int
    ffn: int
    layers: int
    heads: int
    kv_heads: int
    vocab: int
    rope_theta: float
    eps: float
    head_dim: int

    @classmethod
    def from_json(cls, path):
        c = json.load(open(path))
        return cls(c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"],
                   c["num_attention_heads"], c["num_key_value_heads"], c["vocab_size"],
                   float(c.get("rope_theta", 10000.0)), float(c["rms_norm_eps"]),
                   c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"])


class Safetensors:
    """Minimal safetensors reader (no torch): memory-mapped, bf16 widened to float32."""

    def __init__(self, path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.meta = json.loads(f.read(n))
        self.meta.pop("__metadata__", None)
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def __contains__(self, name):
        return name in self.meta

    def get(self, name):
        m = self.meta[name]
        a, b = m["data_offsets"]
        raw = self.mm[self.base + a:self.base + b]
        dt = m["dtype"]
        if dt == "U8":
            return np.asarray(raw).reshape(m["shape"])
        if dt == "BF16":
            u = np.asarray(raw).view("<u2").astype(np.uint32) << 16
            return u.view(np.float32).reshape(m["shape"])
        if dt == "F32":
            return np.asarray(raw).view("<f4").reshape(m["shape"])
        if dt in ("F16", "U32"):
            return np.asarray(raw).view("<f2" if dt == "F16" else "<u4").reshape(m["shape"])
        raise ValueError(f"{name}: dtype {dt}")


def rmsnorm(x, w, eps):
    x = x.astype(np.float32)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


class RefBackend:
    """HF AutoBitLinear semantics in numpy."""

    def __init__(self):
        self.w = {}

    def add(self, name, packed, scale):
        self.w[name] = (packed, float(scale))

    def linear(self, name, x):
        packed, s = self.w[name]
        return reference_linear(x, packed, s)


class TmmBackend:
    """Projections as TMM programs. `run(prog, dram_in)` is supplied by the ISA / board subclass."""

    X_AT = 0x10000
    OUT_AT = 0x100000

    def __init__(self, cfg):
        self.cfg = cfg
        self.w = {}
        self.cycles = 0
        self.runs = 0

    def add(self, name, packed, scale):
        tw = hf_to_tmm(packed, self.cfg.D)
        n = packed.shape[0] * 4
        self.w[name] = self._store(name, tw, np.full(n, scale, np.float32), packed.shape[1])

    def linear(self, name, x):
        w_at, s_at, n, k, rs = self.w[name]
        x = np.ascontiguousarray(x, np.float32)
        d = self.cfg.D
        kb = k // d
        mmax = min(getattr(self.cfg, "ACT_ROWS", 0) or 16, x.shape[0],
                   self.cfg.TMEM_WORDS // k)
        assert mmax >= 1, "one activation row does not fit TMEM"
        out = np.empty((x.shape[0], n), np.float32)
        for r0 in range(0, x.shape[0], mmax):
            xm = x[r0:r0 + mmax]
            m = xm.shape[0]
            nt = min(n, self.cfg.TMEM_WORDS // m)      # QACT has consumed x: results reuse TMEM from 0
            prog = [I.ld(self.X_AT, 0, m * k), I.qact(0, m, 0, kb, k)]
            tiles = []
            for n0 in range(0, n, nt):
                c = min(nt, n - n0)
                o_at = self.OUT_AT + sum(t[2] * m * 4 for t in tiles)
                prog += [I.tmm(w_at + n0 * rs, s_at + n0 * 4, 0, c, kb, rs, c, m, 0, 4, rsc=True),
                         I.st(o_at, 0, m * c)]
                tiles.append((n0, c, c))
            prog.append(I.halt())
            res = self._run(prog, xm, sum(t[2] for t in tiles) * m * 4)
            off = 0
            for n0, c, _ in tiles:
                out[r0:r0 + m, n0:n0 + c] = res[off:off + m * c].reshape(m, c)
                off += m * c
        return out


class IsaBackend(TmmBackend):
    def __init__(self, cfg, dram_bytes=1 << 24):
        super().__init__(cfg)
        self.dram = np.zeros(dram_bytes, np.uint8)
        self.top = 0x200000

    def _store(self, name, tw, sc, k):
        w_at = self.top
        self.dram[w_at:w_at + tw.size] = tw.reshape(-1)
        s_at = (w_at + tw.size + 63) // 64 * 64
        self.dram[s_at:s_at + sc.nbytes] = sc.view(np.uint8)
        self.top = (s_at + sc.nbytes + 63) // 64 * 64
        return w_at, s_at, sc.size, k, tw.shape[1]

    def _run(self, prog, xm, nbytes):
        from opentpu.isasim import Machine
        self.dram[self.X_AT:self.X_AT + xm.nbytes] = xm.view(np.uint8).reshape(-1)
        sl = Machine(self.cfg, [prog], [self.dram.copy()]).run().slices[0]
        self.runs += 1
        return sl.dram[self.OUT_AT:self.OUT_AT + nbytes].copy().view("<f4")


class BoardBackend(TmmBackend):
    """Weights live in the card's DRAM from W_BASE up; one TMM program per call."""

    W_BASE = 0x1000000
    PROG_AT = 0x3C0000

    def __init__(self, board, cfg):
        super().__init__(cfg)
        self.board = board
        self.top = self.W_BASE

    def _store(self, name, tw, sc, k):
        w_at = self.top
        self.board.write(w_at, tw.reshape(-1))
        s_at = (w_at + tw.size + 63) // 64 * 64
        self.board.write(s_at, sc.view(np.uint8))
        self.top = (s_at + sc.nbytes + 63) // 64 * 64
        assert self.top <= self.cfg.DRAM_BYTES, "weights exceed card DRAM"
        return w_at, s_at, sc.size, k, tw.shape[1]

    def _run(self, prog, xm, nbytes):
        self.board.write(self.X_AT, xm.view(np.uint8).reshape(-1))
        self.board.load(self.PROG_AT, prog)
        st = self.board.run(timeout=30.0)
        self.cycles += st["cycles"]
        self.runs += 1
        return self.board.read(self.OUT_AT, nbytes).view("<f4")


class BitNetModel:
    def __init__(self, cfg: BitNetConfig, st, backend, layers=None):
        self.cfg, self.st, self.be = cfg, st, backend
        self.layers = cfg.layers if layers is None else layers
        self.embed = st.get("model.embed_tokens.weight")
        self.norm = st.get("model.norm.weight")
        self.ln = []
        for i in range(self.layers):
            p = f"model.layers.{i}."
            self.ln.append({k: st.get(p + k + ".weight") for k in (
                "input_layernorm", "post_attention_layernorm",
                "self_attn.attn_sub_norm", "mlp.ffn_sub_norm")})
            for pj in PROJ:
                sub = "self_attn." if pj in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp."
                n = p + sub + pj
                backend.add(n, st.get(n + ".weight"), st.get(n + ".weight_scale").reshape(-1)[0])
        hd = cfg.head_dim
        inv = cfg.rope_theta ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)
        self.inv_freq = inv
        self.cache = [None] * self.layers
        self.pos = 0

    def reset(self):
        self.cache = [None] * self.layers
        self.pos = 0

    def _rope(self, x, pos):
        ang = pos[:, None] * self.inv_freq[None, :]
        cos = np.cos(np.concatenate([ang, ang], -1)).astype(np.float32)[:, None, :]
        sin = np.sin(np.concatenate([ang, ang], -1)).astype(np.float32)[:, None, :]
        h = x.shape[-1] // 2
        rot = np.concatenate([-x[..., h:], x[..., :h]], -1)
        return x * cos + rot * sin

    def forward(self, tokens):
        """Feed `tokens` at the current position; returns the logits of the last one."""
        c, be = self.cfg, self.be
        t = len(tokens)
        pos = np.arange(self.pos, self.pos + t)
        h = self.embed[np.asarray(tokens)].astype(np.float32)
        rep = c.heads // c.kv_heads
        for i in range(self.layers):
            p, ln = f"model.layers.{i}.", self.ln[i]
            x = rmsnorm(h, ln["input_layernorm"], c.eps)
            q = be.linear(p + "self_attn.q_proj", x).reshape(t, c.heads, c.head_dim)
            k = be.linear(p + "self_attn.k_proj", x).reshape(t, c.kv_heads, c.head_dim)
            v = be.linear(p + "self_attn.v_proj", x).reshape(t, c.kv_heads, c.head_dim)
            q, k = self._rope(q, pos), self._rope(k, pos)
            if self.cache[i] is not None:
                k = np.concatenate([self.cache[i][0], k])
                v = np.concatenate([self.cache[i][1], v])
            self.cache[i] = (k, v)
            kk, vv = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
            sc = np.einsum("thd,shd->hts", q, kk) * c.head_dim ** -0.5
            mask = np.arange(kk.shape[0])[None, :] > pos[:, None]
            sc = np.where(mask[None], -np.inf, sc)
            sc = np.exp(sc - sc.max(-1, keepdims=True))
            sc /= sc.sum(-1, keepdims=True)
            a = np.einsum("hts,shd->thd", sc, vv).reshape(t, -1)
            a = rmsnorm(a, ln["self_attn.attn_sub_norm"], c.eps)
            h = h + be.linear(p + "self_attn.o_proj", a)
            x = rmsnorm(h, ln["post_attention_layernorm"], c.eps)
            g = np.maximum(be.linear(p + "mlp.gate_proj", x), 0) ** 2
            u = be.linear(p + "mlp.up_proj", x)
            m = rmsnorm(g * u, ln["mlp.ffn_sub_norm"], c.eps)
            h = h + be.linear(p + "mlp.down_proj", m)
        self.pos += t
        last = rmsnorm(h[-1], self.norm, c.eps)
        return self.embed @ last

    def generate(self, prompt, n, eos=None):
        self.reset()
        logits = self.forward(prompt)
        out = []
        for _ in range(n):
            tok = int(np.argmax(logits))
            out.append(tok)
            if tok == eos:
                break
            logits = self.forward([tok])
        return out
