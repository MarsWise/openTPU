"""LeRobot's SmolVLA as a resident graph (opentpu/resident.py): the SigLIP vision tower, the
connector (pixel shuffle + projection), the language embedding rows, the state projection, the
SmolVLM text layers filling each layer's prefix K / V, then every Euler step of the action
expert (suffix embedding with the sinusoidal time embedding, self-attention layers over the
prefix K / V and the actions' own (causal), cross-attention layers over the expert's
projections of the prefix K / V, the action head and the update) run on the device over
tensors in the card's DRAM. The host writes the preprocessed image(s) (the policy's
resize / pad / [-1, 1] normalization), the state, the noise and the language token ids'
embedding rows' addresses (the tokenizer's ids: a device copy of each table row), launches the
programs and reads the actions. With device_noise (Config.VRAND) the host writes a seed and the
device draws the flow's initial noise (kernels/randn.py; VLA.noise(seed) the same values on the
host, bit for bit): FPGA-only.

build(policy, cfg, n_lang) -> VLA; VLA.image(...), VLA.programs(), VLA.actions(img).
"""
from __future__ import annotations

import math

import numpy as np

from . import resident as RG
from .compiler import Tensor


def _np(t):
    return t.detach().float().cpu().numpy()


def _b(lin):
    return None if lin.bias is None else _np(lin.bias)


def _shuffle_map(seq: int, C: int, s: int) -> np.ndarray:
    """SmolVLM's pixel_shuffle as an index map: out[t, j * C:(j + 1) * C] = x[map[t, j]]."""
    side = int(seq ** 0.5)
    idx = np.arange(seq).reshape(side, side)
    x = np.repeat(idx[:, :, None], C, 2)                       # (h, w, C): each row's index
    x = x.reshape(side, side // s, C * s).transpose(1, 0, 2)
    x = x.reshape(side // s, side // s, C * s * s).transpose(1, 0, 2)
    x = x.reshape(seq // (s * s), C * s * s)
    return x[:, ::C]                                            # [tokens, s*s] source rows


class VLA:
    def __init__(self, g, imgs, lang_rows, state, x, enc_ops, step_ops, steps, chunk, A):
        self.g, self.imgs, self.lang_rows, self.state, self.x = g, imgs, lang_rows, state, x
        self.enc = g.compile(enc_ops)
        self.step = [g.compile(o) for o in step_ops]
        self.steps, self.chunk, self.A = steps, chunk, A
        self.seed = None                       # device_noise: the seed's word

    def noise(self, seed: int) -> np.ndarray:
        """device_noise: the initial noise [chunk, A] the device draws from `seed`."""
        from .kernels.randn import seeded_reference
        return seeded_reference(self.chunk, self.A, seed, 0)

    def programs(self) -> list:
        out = list(self.enc)
        for s in self.step:
            out += s
        return out

    def image(self, images, state, noise, seed: int | None = None) -> np.ndarray:
        """images: [C, H, W] each (the policy's preprocessed pixels); state [max_state_dim];
        noise [chunk, max_action_dim], or with device_noise None and the `seed`."""
        g = self.g
        img = g.image()
        f = img.view(np.float32)
        for t, x in zip(self.imgs, images):
            x = np.asarray(x, np.float32).transpose(1, 2, 0)
            for h in range(t.H):
                o = (t.addr + 4 * ((t.border + h) * t.rs + t.border * t.C)) // 4
                f[o:o + t.W * t.C] = x[h].reshape(-1)
        g.put(img, self.state, np.asarray(state, np.float32).reshape(1, -1))
        if self.seed is not None:
            if seed is None or noise is not None:
                raise ValueError("device_noise: a seed, and no host noise")
            img.view(np.uint32)[self.seed.addr // 4] = seed & 0xFFFFFFFF     # (raw)
            return img
        g.put(img, self.x, np.asarray(noise, np.float32).reshape(self.chunk, self.A))
        return img

    def actions(self, img) -> np.ndarray:
        return self.g.read(img, self.x)


def build(policy, cfg, lang_tokens, steps: int | None = None,
          device_noise: bool = False) -> VLA:
    """policy: a LeRobot SmolVLAPolicy (B = 1, every image and language token valid: no
    padding); lang_tokens: the instruction's ids (their embedding rows are copied on the
    device, the ids are compiled into the copies' addresses). device_noise (Config.VRAND): the
    initial noise drawn on the device from a seed (VLA.image's seed)."""
    m = policy.model
    pc = m.config
    vwe = m.vlm_with_expert
    vlm = vwe.get_vlm_model()
    vis, conn, text, expert = vlm.vision_model, vlm.connector, vlm.text_model, vwe.lm_expert
    steps = steps or pc.num_steps
    chunk, A = pc.chunk_size, pc.max_action_dim
    g = RG.Graph(cfg)
    D = g.D
    Ct = text.config.hidden_size
    H, KVH, dh = (text.config.num_attention_heads, text.config.num_key_value_heads,
                  text.config.head_dim)

    # ---- prefix: images
    ve = vis.embeddings
    p = ve.patch_size
    imgs, img_tok = [], []
    for ci, _ in enumerate(pc.image_features):
        _, Hi, Wi = (3,) + tuple(reversed(pc.resize_imgs_with_padding))
        xt = g.tensor2(Hi, Wi, 3, 0, D)
        imgs.append(xt)
        pe = g.conv2d(xt, _np(ve.patch_embedding.weight), _np(ve.patch_embedding.bias), p, 0,
                      name=f"img{ci}.patch")
        Np, Cv = pe.H * pe.W, pe.C
        h = RG.DT(pe.addr, Np, Cv)
        pos = g.tensor(Np, Cv)
        g.consts.append((pos.addr, _np(ve.position_embedding.weight)[:Np].copy()))
        h = g.add(h, pos, name=f"img{ci}.pos")
        for li, L in enumerate(vis.encoder.layers):
            n = f"img{ci}.v{li}"
            a = g.layer_norm(h, _np(L.layer_norm1.weight), _np(L.layer_norm1.bias),
                             L.layer_norm1.eps, name=n + ".ln1")
            at = L.self_attn
            vh = at.num_heads
            vd = Cv // vh
            q = g.linear(a, _np(at.q_proj.weight), _b(at.q_proj), name=n + ".q")
            k = g.linear(a, _np(at.k_proj.weight), _b(at.k_proj), name=n + ".k")
            v = g.linear(a, _np(at.v_proj.weight), _b(at.v_proj), name=n + ".v")
            o = g.attention(q, k, v, vh, vh, vd, [(0, Np, Np)], scale=at.scale, name=n + ".attn")
            o = g.linear(o, _np(at.out_proj.weight), _b(at.out_proj), name=n + ".o")
            h = g.add(h, o, name=n + ".res1")
            a = g.layer_norm(h, _np(L.layer_norm2.weight), _np(L.layer_norm2.bias),
                             L.layer_norm2.eps, name=n + ".ln2")
            f1 = g.linear(a, _np(L.mlp.fc1.weight), _b(L.mlp.fc1), act="gelu_tanh",
                          name=n + ".fc1")
            f2 = g.linear(f1, _np(L.mlp.fc2.weight), _b(L.mlp.fc2), name=n + ".fc2")
            h = g.add(h, f2, name=n + ".res2")
        pl = vis.post_layernorm
        h = g.layer_norm(h, _np(pl.weight), _np(pl.bias), pl.eps, name=f"img{ci}.post")
        s = conn.scale_factor
        mp = _shuffle_map(Np, Cv, s)
        sh = g.tensor(mp.shape[0], Cv * s * s)
        g.copy_rows(h.rows(), sh.rows(), [(mp[t, j], t, j * Cv) for t in range(mp.shape[0])
                                          for j in range(mp.shape[1])], Cv,
                    name=f"img{ci}.shuffle")     # (a layout: row moves on the device)
        proj = conn.modality_projection
        proj = getattr(proj, "proj", proj)
        e = g.linear(sh, _np(proj.weight), _b(proj), name=f"img{ci}.proj")
        img_tok.append(g.scale(e, math.sqrt(Ct), name=f"img{ci}.scale"))

    # ---- prefix rows: [image start?] image tokens [image end?] ... language, state
    emb = _np(text.get_input_embeddings().weight)
    special = []
    if m.add_image_special_tokens:
        special = [int(m.global_image_start_token.reshape(-1)[0]),
                   int(m.image_end_token.reshape(-1)[0])]
    rows = []                                  # (kind, payload)
    for t in img_tok:
        if special:
            rows.append(("tok", special[0], 1.0))
        rows += [("img", (t, i)) for i in range(t.L)]
        if special:
            rows.append(("tok", special[1], 1.0))
    rows += [("tok", int(i), math.sqrt(Ct)) for i in np.asarray(lang_tokens).reshape(-1)]
    rows.append(("state",))
    P = len(rows)
    x = g.tensor(P, Ct)
    table = g.const(emb.astype(np.float32))
    lang_rows = []
    tabt = Tensor(table, (emb.shape[0], Ct), (Ct, 1))
    tok_moves, img_moves = [], {}
    for r, item in enumerate(rows):
        if item[0] == "img":
            t, i = item[1]
            img_moves.setdefault(id(t), (t, []))[1].append((i, r, 0))
        elif item[0] == "tok":
            tok_moves.append((item[1], r, 0))
            if item[2] != 1.0:
                lang_rows.append(r)
    g.copy_rows(tabt, x.rows(), tok_moves, Ct, name="prefix.tok")
    for t, mv in img_moves.values():
        g.copy_rows(t.rows(), x.rows(), mv, Ct, name="prefix.img")
    for r in lang_rows:                        # language embeddings * sqrt(hidden)
        rr = RG.DT(x.addr + 4 * r * Ct, 1, Ct)
        g.scale(rr, math.sqrt(Ct), name="prefix.lang_scale")
    sd = m.state_proj.in_features
    state = g.tensor(1, sd)
    g.linear(state, _np(m.state_proj.weight), _b(m.state_proj),
             y=RG.DT(x.addr + 4 * (P - 1) * Ct, 1, Ct), name="prefix.state")

    # ---- the text layers over the prefix: each layer's K / V (rows 0 .. P - 1 of buffers that
    # also hold the actions' own K / V in the expert's self-attention layers)
    layers = vwe.get_model_layers([text, expert])
    nL = len(layers[0])
    Kc = [g.tensor(P + chunk, KVH * dh) for _ in range(nL)]
    Vc = [g.tensor(P + chunk, KVH * dh) for _ in range(nL)]
    pos = np.arange(P)
    groups = [(0, P - 1, P - 1), (P - 1, P, P)]          # prefix-LM: the state sees all
    for li, L in enumerate(layers[0]):
        n = f"vlm{li}"
        a = g.rms_norm(x, _np(L.input_layernorm.weight), L.input_layernorm.variance_epsilon,
                       name=n + ".norm1")
        at = L.self_attn
        q = g.linear(a, _np(at.q_proj.weight), _b(at.q_proj), name=n + ".q")
        k = RG.DT(Kc[li].addr, P, KVH * dh)
        v = RG.DT(Vc[li].addr, P, KVH * dh)
        g.linear(a, _np(at.k_proj.weight), _b(at.k_proj), y=k, name=n + ".k")
        g.linear(a, _np(at.v_proj.weight), _b(at.v_proj), y=v, name=n + ".v")
        g.rope(q, H, dh, pos, name=n + ".rope_q")
        g.rope(k, KVH, dh, pos, name=n + ".rope_k")
        if li == nL - 1:
            break                              # (the prefix's last outputs are not used)
        o = g.attention(q, k, v, H, KVH, dh, groups, name=n + ".attn")
        o = g.linear(o, _np(at.o_proj.weight), _b(at.o_proj), name=n + ".o")
        x = g.add(x, o, name=n + ".res1")
        a = g.rms_norm(x, _np(L.post_attention_layernorm.weight),
                       L.post_attention_layernorm.variance_epsilon, name=n + ".norm2")
        mlp = L.mlp
        gt = g.linear(a, _np(mlp.gate_proj.weight), _b(mlp.gate_proj), act="silu",
                      name=n + ".gate")
        up = g.linear(a, _np(mlp.up_proj.weight), _b(mlp.up_proj), name=n + ".up")
        dn = g.linear(g.mul(gt, up, name=n + ".glu"), _np(mlp.down_proj.weight),
                      _b(mlp.down_proj), name=n + ".down")
        x = g.add(x, dn, name=n + ".res2")
    enc_ops, g.ops = g.ops, []

    # the cross-attention layers' expert K / V of the prefix: fixed for the whole loop
    cross = {}
    for li, EL in enumerate(layers[1]):
        if EL is None:
            continue
        self_attn = ("cross" not in vwe.attention_mode or
                     (vwe.self_attn_every_n_layers > 0 and li % vwe.self_attn_every_n_layers == 0))
        if self_attn:
            continue
        kp = RG.DT(Kc[li].addr, P, KVH * dh)
        vp = RG.DT(Vc[li].addr, P, KVH * dh)
        ek = g.linear(kp, _np(EL.self_attn.k_proj.weight), _b(EL.self_attn.k_proj),
                      name=f"exp{li}.cross_k")
        ev = g.linear(vp, _np(EL.self_attn.v_proj.weight), _b(EL.self_attn.v_proj),
                      name=f"exp{li}.cross_v")
        cross[li] = (ek, ev)
    enc_ops += g.ops
    g.ops = []

    # ---- the Euler loop
    Ce = expert.config.hidden_size
    eH, eKVH = expert.config.num_attention_heads, expert.config.num_key_value_heads
    edh = expert.config.head_dim
    xt = g.tensor(chunk, A)
    seed = None
    if device_noise:
        from .kernels.randn import seeded_normal_rows
        from .rand import counters
        if not getattr(cfg, "VRAND", False):
            raise ValueError("device_noise needs Config.VRAND (VOP RAND)")
        seed = g.tensor(1, 1)
        enc_ops.append(("noise", seeded_normal_rows,
                        dict(c=g.vec(counters(chunk * A)), seed=seed.rows(), y=xt.rows(),
                             M=chunk, N=A, key=0)))
    dt = -1.0 / steps
    frac = np.linspace(0.0, 1.0, Ce // 2, dtype=np.float64)
    period = pc.min_period * (pc.max_period / pc.min_period) ** frac
    freq = g.vec((1.0 / period * 2 * math.pi).astype(np.float32))
    times = g.tensor(steps, 1)
    g.consts.append((times.addr, np.array([[1.0 + s * dt] for s in range(steps)], np.float32)))
    temb = g.tensor(1, Ce)
    W_in = _np(m.action_time_mlp_in.weight)                  # [Ce, 2 Ce]: (action, time)
    step_ops = []
    from .kernels import eltwise as E
    for s in range(steps):
        g.ops = []
        g.op("time_emb", E.sincos_embed, x=times.rows(s, s + 1), f=freq, y=temb.rows(), M=1,
             N=Ce)
        tb = g.linear(temb, W_in[:, Ce:], _b(m.action_time_mlp_in), name="time_mlp")
        ae = g.linear(xt, _np(m.action_in_proj.weight), _b(m.action_in_proj), name="action_in")
        ai = g.linear(ae, W_in[:, :Ce], None, name="mlp_in")
        g.add_row(ai, Tensor(tb.addr, (Ce,), (1,)), name="mlp_in.time")
        g.unary(ai, "silu", y=ai, name="mlp_in.silu")
        h = g.linear(ai, _np(m.action_time_mlp_out.weight), _b(m.action_time_mlp_out),
                     name="mlp_out")
        spos = P + np.arange(chunk)
        for li, EL in enumerate(layers[1]):
            if EL is None:
                continue
            n = f"exp{li}"
            a = g.rms_norm(h, _np(EL.input_layernorm.weight),
                           EL.input_layernorm.variance_epsilon, name=n + ".norm1")
            at = EL.self_attn
            q = g.linear(a, _np(at.q_proj.weight), _b(at.q_proj), name=n + ".q")
            if li in cross:
                g.rope(q, eH, edh, np.arange(chunk), name=n + ".rope_q")
                ek, ev = cross[li]
                o = g.attention(q, ek, ev, eH, ek.C // edh, edh, [(0, chunk, P)],
                                scale=dh ** -0.5, name=n + ".xattn")
            else:
                k = RG.DT(Kc[li].addr + 4 * P * KVH * dh, chunk, KVH * dh)
                v = RG.DT(Vc[li].addr + 4 * P * KVH * dh, chunk, KVH * dh)
                g.linear(a, _np(at.k_proj.weight), _b(at.k_proj), y=k, name=n + ".k")
                g.linear(a, _np(at.v_proj.weight), _b(at.v_proj), y=v, name=n + ".v")
                g.rope(q, eH, edh, spos, name=n + ".rope_q")
                g.rope(k, eKVH, edh, spos, name=n + ".rope_k")
                o = g.attention(q, Kc[li], Vc[li], eH, eKVH, edh,
                                [(r, r + 1, P + r + 1) for r in range(chunk)], per_prog=8,
                                scale=dh ** -0.5, name=n + ".attn")
            o = g.linear(o, _np(at.o_proj.weight), _b(at.o_proj), name=n + ".o")
            h = g.add(h, o, name=n + ".res1")
            a = g.rms_norm(h, _np(EL.post_attention_layernorm.weight),
                           EL.post_attention_layernorm.variance_epsilon, name=n + ".norm2")
            mlp = EL.mlp
            gt = g.linear(a, _np(mlp.gate_proj.weight), _b(mlp.gate_proj), act="silu",
                          name=n + ".gate")
            up = g.linear(a, _np(mlp.up_proj.weight), _b(mlp.up_proj), name=n + ".up")
            dn = g.linear(g.mul(gt, up, name=n + ".glu"), _np(mlp.down_proj.weight),
                          _b(mlp.down_proj), name=n + ".down")
            h = g.add(h, dn, name=n + ".res2")
        h = g.rms_norm(h, _np(expert.norm.weight), expert.norm.variance_epsilon,
                       name="exp.norm")
        vt = g.linear(h, _np(m.action_out_proj.weight), _b(m.action_out_proj),
                      name="action_out")
        g.axpy(xt, vt, dt, name="euler")
        step_ops.append(g.ops)
    g.ops = []
    vla = VLA(g, imgs, None, state, xt, enc_ops, step_ops, steps, chunk, A)
    vla.seed = seed
    return vla
