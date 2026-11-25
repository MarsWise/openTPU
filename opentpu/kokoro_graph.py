"""Kokoro-82M v1.0 text to speech on the device: every operation of KModel.forward_with_tokens as
resident-graph programs (opentpu/resident.py), in two graphs:

  A  (the input ids and the voice's style row written by the host)
     PL-BERT (ALBERT, 12 passes of one shared layer) -> bert_encoder -> the duration encoder
     (3 x [BiLSTM, AdaLayerNorm]) -> BiLSTM -> duration_proj -> sigmoid, summed: the
     durations; the text encoder (embedding, 3 x [conv5, LayerNorm, LeakyReLU], BiLSTM)
  B  (after the host reads the T durations back and rounds them: the frame count fixes graph
     B's shapes, the alignment's row moves are its copy programs)
     en / asr = the durations' row repeats of d / t_en (row copies on the device)
     F0Ntrain (BiLSTM, 2 x 3 AdainResBlk1d, 1x1 projections) -> F0, N
     Decoder (F0 / N convs, encode, 4 decode blocks) -> Generator: SineGen (harmonics, phase
     cumulative sums, linear upsampling, sin, Gaussian noise from VOP RAND), the STFT of the
     source (fp32 DFT, atan2), 2 x [ConvTranspose, noise conv + AdaINResBlock1, 3
     AdaINResBlock1 averaged], conv_post, iSTFT -> PCM at 24 kHz

The host's part: phonemes -> ids (the vocabulary table), the style row (the voice pack's row
for the phoneme count), reading the T durations and rounding them (a control decision: graph
B's shape), the noise seed. Nothing of the model's arithmetic runs on the host.

Weight norm is folded once at conversion (fold_weight_norm); depthwise transposed convolutions
(AdainResBlk1d's pool) run as dense block-diagonal ones; nearest upsampling and concatenations
are row / column copies. The reference: tools/kokoro/ref (hexgrad's kokoro 0.9.4 modules).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import language as ol
from . import resident as RG
from .compiler import Tensor
from .kernels import atan2 as AT
from .kernels import lib
from .kernels import randn as RN
from .kokoro_resident import adain, adain_resblock, bilstm, decoder_tail, fold_weight_norm, \
    dilated_conv

SR = 24000
UPS = 300                                  # samples a frame (prod(upsample_rates) * hop)
NFFT, HOP = 20, 5
HARM = 9                                   # SineGen's harmonics (harmonic_num 8 + 1)
MAGIC = float(1.5 * 2 ** 23)               # x + MAGIC - MAGIC: x rounded to an integer (fp32)


# ================================================================== small kernels
@ol.jit
def leaky_k(x, y, slope: float):
    v = ol.load(x)
    ol.store(y, ol.maximum(v, v * slope))


@ol.jit
def add_scale_k(a, b, y, s: float):
    """y = (a + b) * s (AdainResBlk1d's (residual + shortcut) / sqrt(2), the resblocks' mean)."""
    t = ol.load(a)
    t.set((t + ol.load(b)) * s)
    ol.store(y, t)


@ol.jit
def add3_scale_k(a, b, c, y, s: float):
    t = ol.load(a)
    t.set((t + ol.load(b) + ol.load(c)) * s)
    ol.store(y, t)


@ol.jit
def dur_k(x, y, T: int):
    """y [T, 1] = sum over 50 of sigmoid(x [T, 50]) (KModel: the durations before rounding)."""
    v = lib.sigmoid(ol.load(x))
    ol.store(y, ol.sum(v, axis=1)[:, None])


@ol.jit
def phase_k(f0r, kk, p, inc, sv, nv, prev, n: int, first: bool):
    """SineGen's per-frame quantities for frames f0r [n, 9] (each frame's F0 in all 9 columns):
    inc = UPS * ((k f0 / SR) mod 1) (k = 1 .. 9; mod 1 up to an integer: x - round(x)), the
    phase P[i] = (P[i - 1] + inc[i]) mod 1 (cycles; P[-1] from `prev` unless first), the voiced
    amplitude sv = 0.1 uv and the noise's nv = 0.003 uv + 0.1/3 (1 - uv), uv = f0 > 10."""
    f = ol.load(f0r)
    x = f * ol.load(kk)[None, :]
    r = x - ((x + MAGIC) - MAGIC)
    inc_t = r * float(UPS)
    ol.store(inc, inc_t)
    uv = ol.greater(f, 10.0)
    ol.store(sv, uv * 0.1)
    ol.store(nv, uv * (0.003 - 0.1 / 3) + 0.1 / 3)
    acc = ol.zeros([1, HARM]) if first else ol.load(prev)
    for i in range(n):
        acc.set(acc + inc_t[i:i + 1, :])
        acc.set(acc - ((acc + MAGIC) - MAGIC))
        ol.store(p[i:i + 1, :], acc)


@ol.jit
def samples_k(p, inc, sv, nv, noise, A, C, w, y, i0: int, m: int, n: int, bias: float):
    """Frames i0 .. i0 + m of SineGen's samples, merged: per frame i, its UPS samples' phase
    P[i] - A inc[i] + C inc[i + 1] (F.interpolate's linear weights, A and C [UPS, 9]; clamped:
    no A term at frame 0, no C term at the last), sin(2 pi (. mod 1)), times sv[i] plus noise
    times nv[i], then tanh(w . harmonics + bias) -> y [m UPS]."""
    At, Ct, wt = ol.load(A), ol.load(C), ol.load(w)
    for i in range(i0, i0 + m):
        ph = ol.zeros([UPS, HARM]) + ol.load(p[i])[None, :]
        if i > 0:
            ph.set(ph - At * ol.load(inc[i])[None, :])
        if i + 1 < n:
            ph.set(ph + Ct * ol.load(inc[i + 1])[None, :])
        ph.set(ph - ((ph + MAGIC) - MAGIC))
        sn = ol.tfn("sin", ph * float(2 * math.pi))
        h = sn * ol.load(sv[i])[None, :] + ol.load(noise[i * UPS:(i + 1) * UPS, :]) * \
            ol.load(nv[i])[None, :]
        out = ol.sum(h * wt[None, :], axis=1) + bias
        ol.store(y[(i - i0) * UPS:(i - i0 + 1) * UPS], lib.tanh(out))


@ol.jit
def stft_k(fr, basis, y, bins: int):
    """y [rows, 2 bins] = |X|, angle(X) of frames fr [rows, NFFT] (the windowed DFT basis,
    transposed: [2 bins, NFFT], the cos rows then the -sin rows; fp32 on the VPU)."""
    f, b = ol.load(fr), ol.load(basis)
    rows = f.shape[0]
    re = ol.empty([rows, bins])
    im = ol.empty([rows, bins])
    for j in range(bins):
        re[:, j:j + 1].set(ol.sum(f * b[j, :][None, :], axis=1)[:, None])
        im[:, j:j + 1].set(ol.sum(f * b[bins + j, :][None, :], axis=1)[:, None])
    q = re * re + im * im
    ol.store(y[:, 0:bins], q * ol.rsqrt(q + 1e-30))
    ol.store(y[:, bins:2 * bins], AT.atan2(im, re))


# ================================================================== the graph builder
@dataclass
class Kokoro:
    """Kokoro-82M's weights (tools/kokoro/convert.py's safetensors: <module>.<key>, fp32) and
    config; build_a / build_b make the two graphs on one Graph (B's tensors after A's)."""
    W: dict
    cfg: dict
    eps_bert: float = 1e-12
    style: object = None
    t: dict = field(default_factory=dict)

    def w(self, k):
        return np.asarray(self.W[k], np.float32)

    def wn(self, p):
        """A weight-normed conv's weight (folded) and bias."""
        b = self.W.get(p + "bias")
        return fold_weight_norm(self.w(p + "weight_v"), self.w(p + "weight_g")), \
            None if b is None else np.asarray(b, np.float32)

    def sub(self, prefix):
        n = len(prefix)
        return {k[n:]: v for k, v in self.W.items() if k.startswith(prefix)}

    # ---- helpers on the graph
    @staticmethod
    def leaky(g, x, slope, name="leaky"):
        y = g.tensor(x.L, x.C, x.pad)
        g._rowwise(name, leaky_k, x.L, x.C,
                   lambda r0, r1: dict(x=x.rows(r0, r1), y=y.rows(r0, r1)), slope=float(slope))
        return y

    @staticmethod
    def add_scale(g, a, b, s, pad=0, name="add_scale"):
        y = g.tensor(a.L, a.C, pad)
        g._rowwise(name, add_scale_k, a.L, a.C,
                   lambda r0, r1: dict(a=a.rows(r0, r1), b=b.rows(r0, r1), y=y.rows(r0, r1)),
                   s=float(s))
        return y

    @staticmethod
    def gather(g, table: RG.DT, rows, dst: RG.DT, c0=0, name="gather"):
        """dst rows t (columns c0..) = table rows rows[t] (copies on the device)."""
        g.copy_rows(table.rows(), dst.rows(), [(int(r), t, c0) for t, r in enumerate(rows)],
                    table.C, name=name)

    @staticmethod
    def row_vec(x: RG.DT, c0: int, c1: int) -> Tensor:
        return Tensor(x.addr + 4 * (x.pad * x.rs + c0), (c1 - c0,), (1,))

    def table(self, g, a):
        a = np.ascontiguousarray(np.asarray(a, np.float32))
        t = g.tensor(a.shape[0], a.shape[1])
        g.consts.append((t.addr, a))
        return t

    def cat_style(self, g, x, s, name):
        """[x | s] over every row (the duration encoder's concatenations with the style)."""
        y = g.tensor(x.L, x.C + s.C)
        g.copy(x, y.cols(0, x.C), name=name + ".x")
        g.copy_rows(s.rows(), y.rows(), [(0, t, x.C) for t in range(x.L)], s.C,
                    name=name + ".s")
        return y

    # ---- PL-BERT (ALBERT)
    def albert(self, g, ids):
        P = "bert."
        T = len(ids)
        H, heads = 768, 12
        emb = g.tensor(T, 128)
        self.gather(g, self.table(g, self.w(P + "embeddings.word_embeddings.weight")), ids, emb,
                    name="bert.word")
        pos = self.table(g, self.w(P + "embeddings.position_embeddings.weight")[:T]
                         + self.w(P + "embeddings.token_type_embeddings.weight")[0][None, :])
        x = g.add(emb, pos, name="bert.pos_type")
        x = g.layer_norm(x, self.w(P + "embeddings.LayerNorm.weight"),
                         self.w(P + "embeddings.LayerNorm.bias"), self.eps_bert, name="bert.ln")
        L = P + "encoder."
        x = g.linear(x, self.w(L + "embedding_hidden_mapping_in.weight"),
                     self.w(L + "embedding_hidden_mapping_in.bias"), name="bert.map_in")
        A = L + "albert_layer_groups.0.albert_layers.0."
        n_layers = 12
        for i in range(n_layers):
            n = f"bert.l{i}"
            q = g.linear(x, self.w(A + "attention.query.weight"), self.w(A + "attention.query.bias"),
                         name=n + ".q")
            k = g.linear(x, self.w(A + "attention.key.weight"), self.w(A + "attention.key.bias"),
                         name=n + ".k")
            v = g.linear(x, self.w(A + "attention.value.weight"), self.w(A + "attention.value.bias"),
                         name=n + ".v")
            o = g.attention(q, k, v, heads, heads, H // heads, [(0, T, T)], name=n + ".attn")
            o = g.linear(o, self.w(A + "attention.dense.weight"), self.w(A + "attention.dense.bias"),
                         name=n + ".dense")
            a = g.add(o, x, name=n + ".res1")
            a = g.layer_norm(a, self.w(A + "attention.LayerNorm.weight"),
                             self.w(A + "attention.LayerNorm.bias"), self.eps_bert, name=n + ".ln1")
            f = g.linear(a, self.w(A + "ffn.weight"), self.w(A + "ffn.bias"), act="gelu_tanh",
                         name=n + ".ffn")
            f = g.linear(f, self.w(A + "ffn_output.weight"), self.w(A + "ffn_output.bias"),
                         name=n + ".ffn_out")
            x = g.add(f, a, name=n + ".res2")
            x = g.layer_norm(x, self.w(A + "full_layer_layer_norm.weight"),
                             self.w(A + "full_layer_layer_norm.bias"), self.eps_bert,
                             name=n + ".ln2")
        return x

    # ---- the duration side
    def ada_layer_norm(self, g, x, s, prefix, name):
        """AdaLayerNorm: layer_norm without affine, then (1 + gamma) x + beta from fc(s) (the + 1
        folded into fc's bias)."""
        fw, fb = self.w(prefix + "fc.weight"), self.w(prefix + "fc.bias").copy()
        C = x.C
        fb[:C] += 1.0
        proj = g.linear(s, fw, fb, name=name + ".fc")
        y = g.layer_norm(x, np.ones(C, np.float32), np.zeros(C, np.float32), 1e-5,
                         name=name + ".ln")
        return g.film(y, self.row_vec(proj, 0, C), self.row_vec(proj, C, 2 * C),
                      name=name + ".film")

    def duration_encoder(self, g, d_en, s):
        P = "predictor.text_encoder.lstms."
        x = self.cat_style(g, d_en, s, "dur.cat0")
        for i in range(3):
            h = bilstm(g, x, self.sub(f"{P}{2 * i}."))
            h = self.ada_layer_norm(g, h, s, f"{P}{2 * i + 1}.", f"dur.aln{i}")
            x = self.cat_style(g, h, s, f"dur.cat{i + 1}")
        return x                                            # d [T, 640]

    def text_encoder(self, g, ids):
        P = "text_encoder."
        T = len(ids)
        x = g.tensor(T, 512, 2)
        self.gather(g, self.table(g, self.w(P + "embedding.weight")), ids, x, name="tenc.emb")
        for i in range(3):
            w, b = self.wn(f"{P}cnn.{i}.0.")
            y = g.conv1d(x, w, b, pad=2, name=f"tenc.conv{i}")
            y = g.layer_norm(y, self.w(f"{P}cnn.{i}.1.gamma"), self.w(f"{P}cnn.{i}.1.beta"),
                             1e-5, name=f"tenc.ln{i}")
            z = g.tensor(T, 512, 2)
            g._rowwise(f"tenc.act{i}", leaky_k, T, 512,
                       lambda r0, r1: dict(x=y.rows(r0, r1), y=z.rows(r0, r1)), slope=0.2)
            x = z
        return bilstm(g, RG.DT(x.addr, x.L, x.C, x.pad, x.rs), self.sub(P + "lstm."))

    def build_a(self, g, ids, style256):
        """Graph A: (durations [T, 1], d [T, 640], t_en [T, 512]); the style rows are inputs."""
        T = len(ids)
        self.ref_s = g.tensor(1, 256)                       # the voice's row: [decoder | prosody]
        s_dec = RG.DT(self.ref_s.addr, 1, 128, 0, 256)
        s_pro = RG.DT(self.ref_s.addr + 4 * 128, 1, 128, 0, 256)
        sp = g.tensor(1, 128)
        g.copy(s_pro, sp, name="style.prosody")
        sd = g.tensor(1, 128)
        g.copy(s_dec, sd, name="style.decoder")
        self.s_pro, self.s_dec = sp, sd
        bert = self.albert(g, ids)
        d_en = g.linear(bert, self.w("bert_encoder.weight"), self.w("bert_encoder.bias"),
                        name="bert_encoder")
        d = self.duration_encoder(g, d_en, sp)
        x = bilstm(g, d, self.sub("predictor.lstm."))
        dp = g.linear(x, self.w("predictor.duration_proj.linear_layer.weight"),
                      self.w("predictor.duration_proj.linear_layer.bias"), name="dur.proj")
        dur = g.tensor(T, 1)
        g.op("dur.sum", dur_k, x=dp.rows(), y=dur.rows(), T=T)
        t_en = self.text_encoder(g, ids)
        self.t.update(bert=bert, d_en=d_en, d=d, dur_x=x, dur=dur, t_en=t_en)
        return dur

    # ---- graph B
    def adain_resblk1d(self, g, x, s, P, name, upsample=False):
        """istftnet.AdainResBlk1d (StyleTTS2's): (residual + shortcut) / sqrt(2)."""
        C = x.C
        st = self.sub(P)
        Cout = st["conv1.weight_v"].shape[0]
        nw = st.get("norm1.norm.weight", np.ones(C, np.float32))
        nb = st.get("norm1.norm.bias", np.zeros(C, np.float32))
        h = adain(g, x, s, st["norm1.fc.weight"], st["norm1.fc.bias"], nw, nb)
        h = self.leaky(g, h, 0.2, name + ".act1")
        if upsample:                                        # depthwise ConvTranspose k3 s2 p1 op1
            pw, pb = self.wn(P + "pool.")                   # [C, 1, 3]
            dense = np.zeros((C, C, 3), np.float32)
            dense[np.arange(C), np.arange(C)] = pw[:, 0, :]
            h = self.conv_t(g, h, dense, pb, 2, 1, 1, name + ".pool")
        hp = g.tensor(h.L, h.C, 1)
        g.copy(h, hp, name=name + ".pad1")
        w1, b1 = self.wn(P + "conv1.")
        h = g.conv1d(hp, w1, b1, pad=1, name=name + ".conv1")
        nw = st.get("norm2.norm.weight", np.ones(Cout, np.float32))
        nb = st.get("norm2.norm.bias", np.zeros(Cout, np.float32))
        h = adain(g, h, s, st["norm2.fc.weight"], st["norm2.fc.bias"], nw, nb)
        h = self.leaky(g, h, 0.2, name + ".act2")
        hp = g.tensor(h.L, h.C, 1)
        g.copy(h, hp, name=name + ".pad2")
        w2, b2 = self.wn(P + "conv2.")
        res = g.conv1d(hp, w2, b2, pad=1, name=name + ".conv2")
        sc = x
        if upsample:                                        # nearest x 2
            sc = g.tensor(2 * x.L, C)
            g.copy_rows(x.rows(), sc.rows(), [(t // 2, t, 0) for t in range(2 * x.L)], C,
                        name=name + ".up")
        if "conv1x1.weight_v" in st:
            w, _ = self.wn(P + "conv1x1.")
            sc = g.conv1d(RG.DT(sc.addr, sc.L, sc.C, sc.pad, sc.rs), w, None,
                          name=name + ".conv1x1")
        return self.add_scale(g, res, sc, 1 / math.sqrt(2), name=name + ".out")

    def conv_t(self, g, x, w, b, stride, pad, out_pad, name):
        """nn.ConvTranspose1d [Cin, Cout, k] with output_padding (a zero row at the end)."""
        Cin, Cout, k = w.shape
        e = k - 1 - pad
        Lu = (x.L - 1) * stride + 1 + out_pad
        up = g.tensor(Lu, Cin, e)
        for i, (r0, r1) in enumerate(g._chunks(x.L, Cin)):
            g.op(f"{name}.zero_insert.{i}", RG.copy_k, x=x.rows(r0, r1),
                 y=Tensor(up.addr + 4 * ((e + r0 * stride) * Cin), (r1 - r0, Cin),
                          (stride * Cin, 1)))
        wf = np.ascontiguousarray(w[:, :, ::-1].transpose(1, 0, 2))
        return g.conv1d(up, wf, b, 1, e, name=name)

    def f0n(self, g, en, s):
        x = bilstm(g, en, self.sub("predictor.shared."))
        outs = []
        for kind in ("F0", "N"):
            h = x
            for j in range(3):
                h = self.adain_resblk1d(g, h, s, f"predictor.{kind}.{j}.", f"{kind}{j}",
                                        upsample=(j == 1))
            pw = self.w(f"predictor.{kind}_proj.weight")
            outs.append(g.conv1d(h, pw, self.w(f"predictor.{kind}_proj.bias"),
                                 name=f"{kind}.proj"))
        return outs

    def build_b(self, g, dur_frames):
        """Graph B for the rounded durations (one int a token): -> audio [samples, 1]."""
        T = len(dur_frames)
        idx = np.repeat(np.arange(T), dur_frames)
        Fr = len(idx)
        d, t_en = self.t["d"], self.t["t_en"]
        en = g.tensor(Fr, d.C)
        self.gather(g, d, idx, en, name="align.en")
        asr = g.tensor(Fr, t_en.C)
        self.gather(g, t_en, idx, asr, name="align.asr")
        F0, N = self.f0n(g, en, self.s_pro)                 # [2Fr, 1]
        self.t.update(en=en, asr=asr, F0=F0, N=N)
        sd = self.s_dec
        Dp = "decoder."
        c = []
        for nm in ("F0_conv", "N_conv"):
            src = F0 if nm == "F0_conv" else N
            sp = g.tensor(src.L, 1, 1)
            g.copy(RG.DT(src.addr, src.L, 1, src.pad, src.rs), sp, name=nm + ".pad")
            w, b = self.wn(Dp + nm + ".")
            c.append(g.conv1d(sp, w, b, stride=2, pad=1, name=nm))      # [Fr, 1]
        x0 = g.tensor(Fr, 514)
        g.copy(asr, x0.cols(0, 512), name="dec.cat_asr")
        g.copy(c[0], x0.cols(512, 513), name="dec.cat_f0")
        g.copy(c[1], x0.cols(513, 514), name="dec.cat_n")
        x = self.adain_resblk1d(g, x0, sd, Dp + "encode.", "dec.encode")
        w, b = self.wn(Dp + "asr_res.0.")
        asr_res = g.conv1d(asr, w, b, name="dec.asr_res")
        for i in range(4):
            xc = g.tensor(x.L, 1090)
            g.copy(x, xc.cols(0, 1024), name=f"dec.cat{i}.x")
            g.copy(asr_res, xc.cols(1024, 1088), name=f"dec.cat{i}.asr")
            g.copy(c[0], xc.cols(1088, 1089), name=f"dec.cat{i}.f0")
            g.copy(c[1], xc.cols(1089, 1090), name=f"dec.cat{i}.n")
            x = self.adain_resblk1d(g, xc, sd, f"{Dp}decode.{i}.", f"dec.d{i}",
                                    upsample=(i == 3))
        self.t["dec_x"] = x                                 # [2Fr, 512]
        return self.generator(g, x, sd, F0)

    # ---- the generator
    def sinegen(self, g, F0, seed_t):
        n = F0.L
        G = "decoder.generator."
        lw = self.w(G + "m_source.l_linear.weight").reshape(-1)
        lb = float(self.w(G + "m_source.l_linear.bias").reshape(-1)[0])
        xs = (np.arange(UPS) + 0.5) / UPS - 0.5
        a = np.where(xs < 0, -xs, 0.0)                     # weight on P[i - 1]
        c = np.where(xs < 0, 0.0, xs)                      # weight on P[i + 1]
        At = Tensor(g.const(np.repeat(a[:, None], HARM, 1).astype(np.float32)), (UPS, HARM),
                    (HARM, 1))
        Ct = Tensor(g.const(np.repeat(c[:, None], HARM, 1).astype(np.float32)), (UPS, HARM),
                    (HARM, 1))
        kk = g.vec(np.arange(1, HARM + 1, dtype=np.float32) / SR)
        f0r = g.tensor(n, HARM)                            # F0 in 9 columns (row copies)
        f0v = Tensor(F0.addr + 4 * F0.pad * F0.rs, (n, 1), (F0.rs, 1))
        g.copy_rows(f0v, f0r.rows(), [(i, i, k) for i in range(n) for k in range(HARM)], 1,
                    name="src.f0rep")
        P, inc, sv, nv = (g.tensor(n, HARM) for _ in range(4))
        step = 256
        for i0 in range(0, n, step):
            m = min(step, n - i0)
            g.op(f"src.phase.{i0 // step}", phase_k, f0r=f0r.rows(i0, i0 + m), kk=kk,
                 p=P.rows(i0, i0 + m), inc=inc.rows(i0, i0 + m), sv=sv.rows(i0, i0 + m),
                 nv=nv.rows(i0, i0 + m), prev=P.rows(max(0, i0 - 1), max(1, i0)), n=m,
                 first=(i0 == 0))
        L = n * UPS
        noise = g.tensor(L, HARM)
        from . import rand as RD
        cnt = g.const(RD.bits(RD.counters(L * HARM)).view(np.float32))
        per = 16                                           # frames a noise program
        for i0 in range(0, n, per):
            m = min(per, n - i0)
            g.op(f"src.noise.{i0 // per}", RN.seeded_normal_rows,
                 c=Tensor(cnt + 4 * i0 * UPS * HARM, (m * UPS * HARM,), (1,)), seed=seed_t,
                 y=Tensor(noise.addr + 4 * (i0 * UPS * HARM), (m, UPS * HARM), (UPS * HARM, 1)),
                 M=m, N=UPS * HARM, key=1)
        har = g.tensor(L, 1, 10)
        wv = g.vec(lw)
        per = 8
        for i0 in range(0, n, per):
            m = min(per, n - i0)
            g.op(f"src.samples.{i0 // per}", samples_k, p=P.rows(), inc=inc.rows(),
                 sv=sv.rows(), nv=nv.rows(), noise=noise.rows(), A=At, C=Ct, w=wv,
                 y=Tensor(har.addr + 4 * (har.pad + i0 * UPS), (m * UPS,), (1,)),
                 i0=i0, m=m, n=n, bias=lb)
        self.t.update(noise=noise, har_src=har, phase=P, inc=inc)
        return har

    def stft(self, g, har):
        """torch.stft(n_fft 20, hop 5, hann, center, reflect) -> [frames, 22] (|X|, angle)."""
        L = har.L
        # reflect padding: rows -1 .. -10 = rows 1 .. 10, rows L .. L+9 = rows L-2 .. L-11
        moves = [(10 - j, j, 0) for j in range(10)] + \
            [(L - 2 - j, L + 10 + j, 0) for j in range(10)]
        base = har.addr + 4 * (har.pad - 10) * har.rs      # the buffer from its first pad row
        src = Tensor(har.addr + 4 * har.pad * har.rs, (L, 1), (1, 1))
        dst = Tensor(base, (L + 20, 1), (1, 1))
        g.copy_rows(src, dst, moves, 1, name="stft.reflect")
        frames = L // HOP + 1
        view = Tensor(base, (frames, NFFT), (HOP, 1))
        win = np.hanning(NFFT + 1)[:-1]
        t = np.arange(NFFT)
        ang = 2 * np.pi * t[:, None] * np.arange(NFFT // 2 + 1)[None, :] / NFFT
        basis = np.concatenate([np.cos(ang), -np.sin(ang)], axis=1) * win[:, None]   # [20, 22]
        bt = Tensor(g.const(np.ascontiguousarray(basis.T.astype(np.float32))), (22, NFFT),
                    (NFFT, 1))
        y = g.tensor(frames, 22)
        rows = 256
        for i, r0 in enumerate(range(0, frames, rows)):
            r1 = min(frames, r0 + rows)
            g.op(f"stft.{i}", stft_k, fr=Tensor(base + 4 * HOP * r0, (r1 - r0, NFFT), (HOP, 1)),
                 basis=bt, y=y.rows(r0, r1), bins=11)
        return y

    def generator(self, g, x, s, F0):
        G = "decoder.generator."
        self.seed = g.tensor(1, 1)
        har_src = self.sinegen(g, F0, self.row_vec(self.seed, 0, 1))
        har = self.stft(g, har_src)                          # [Lup/5 + 1, 22]
        self.t["har"] = har
        ups = [(10, 20), (6, 12)]
        for i, (u, k) in enumerate(ups):
            x = self.leaky(g, x, 0.1, f"gen.act{i}")
            if i == 0:
                hp = g.tensor(har.L, 22, 3)
                g.copy(har, hp, name="gen.har_pad")
                nc = g.conv1d(hp, self.w(f"{G}noise_convs.0.weight"),
                              self.w(f"{G}noise_convs.0.bias"), stride=6, pad=3,
                              name="gen.noise_conv0")
            else:
                nc = g.conv1d(har, self.w(f"{G}noise_convs.1.weight"),
                              self.w(f"{G}noise_convs.1.bias"), name="gen.noise_conv1")
            xs = self.resblock1(g, nc, s, f"{G}noise_res.{i}.", f"gen.nres{i}")
            w, b = self.wn(f"{G}ups.{i}.")
            x = self.conv_t(g, x, w, b, u, (k - u) // 2, 0, f"gen.up{i}")
            if i == 1:                                       # ReflectionPad1d((1, 0))
                xr = g.tensor(x.L + 1, x.C)
                g.copy_rows(x.rows(), xr.rows(), [(1, 0, 0)], x.C, name="gen.reflect")
                g.copy(x, RG.DT(xr.addr + 4 * xr.C, x.L, x.C, 0, xr.C), name="gen.reflect.x")
                x = xr
            x = g.add(x, xs, name=f"gen.add{i}")
            outs = [self.resblock1(g, x, s, f"{G}resblocks.{3 * i + j}.", f"gen.rb{i}{j}")
                    for j in range(3)]
            y = g.tensor(x.L, x.C)
            g._rowwise(f"gen.mean{i}", add3_scale_k, x.L, x.C,
                       lambda r0, r1, o=outs, y=y: dict(a=o[0].rows(r0, r1), b=o[1].rows(r0, r1),
                                                        c=o[2].rows(r0, r1), y=y.rows(r0, r1)),
                       tiles=6, s=1.0 / 3)         # a, b, c and a temporary live at once
            x = y
        self.t["gen_x"] = x
        return decoder_tail(g, x, self.w(G + "conv_post.weight_v"),
                            self.w(G + "conv_post.weight_g"), self.w(G + "conv_post.bias"))

    def resblock1(self, g, x, s, P, name):
        st = self.sub(P)
        y = adain_resblock(g, x, s, st)
        return y


def ids_of(vocab: dict, phonemes: str) -> list:
    """KModel.forward's ids: the phonemes in the vocabulary, framed by 0."""
    return [0] + [vocab[p] for p in phonemes if p in vocab] + [0]


def round_durations(dur: np.ndarray, speed: float = 1.0) -> np.ndarray:
    """torch.round(duration / speed).clamp(min=1) (round half to even, as torch)."""
    return np.maximum(np.round(np.asarray(dur, np.float64).reshape(-1) / speed), 1).astype(int)


__all__ = ["Kokoro", "ids_of", "round_durations", "dilated_conv"]
