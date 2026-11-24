"""MTP decoding of native Bonsai 2 (opentpu/llm/mtp.py, docs/mtp.md 9): a tiny native model
(ternary group-128 projections in the Hadamard basis, explicit signs; the tied head GSC) with an
MTP layer in the model's basis, as Ternary-Bonsai-2-4B-PQ2_0-MTP-Q8_0.gguf's Q8_0 nextn layer
is: the MTP layer, fc and draft head int8 / fp4 without a Hadamard, the verify's two rows
through the rows kernel with every DeltaNet out-projection group flushed whole (its Hadamard
block spans the group). Greedy tokens are plain greedy decode's whatever the drafts, the
committed states are plain decode's, and the MTP program follows its numpy reference."""
import numpy as np
import pytest
from dataclasses import replace

from opentpu.isasim import Config
from opentpu.llm.bonsai2 import hadamard
from opentpu.llm.mtp import MTPDecoder, MTPStats, mtp_engine
from opentpu.llm.qwen3 import Engine
from opentpu.llm.qwen35 import Spec, mtp_reference, reference_logits

from test_mtp import _states

pytest.importorskip("torch")

CFG = Config(D=128, MCOLS=4, ACT_ROWS=8, TGSC=True, DRAM_BYTES=32 << 20, DSTEP=True,
             STREAM=True, TFN=True, PAIR=True, TQUAD=True, TPAIR=True, TGRQ=True)
KW = dict(cap=256, cfg=CFG, wformat="ternary_gsc")


def native_tiny(seed=333):
    """(spec, weights as GGUFWeights gives them, the model-basis weights of the reference)."""
    spec = Spec(512, ("linear", "attn"), 4, 2, 256, 64, 8, 128, 128, 1024, 512,
                lin_kheads=4, native_bonsai=True, pair_loop=True)
    rng = np.random.default_rng(seed)
    signs = {k: rng.choice(np.array([-1, 1], np.float32), k) for k in (512, 1024)}

    class W(dict):
        def signs_for(self, width):
            return signs[width]

        def embedding_table(self):
            return hadamard(self["model.embed_tokens.weight"], 512, signs[512], True)

        def head_rows(self, ids):
            return self.embedding_table()[np.asarray(ids)]

    w, eff = W(), {}

    def mat(name, n, k, scale=(.05, .1)):     # (tokens that vary)
        g = rng.uniform(*scale, (n, k // 128)).astype(np.float16).astype(np.float32)
        v = (rng.integers(-1, 2, (n, k), dtype=np.int8).reshape(n, -1, 128)
             * g[..., None]).reshape(n, k)
        w[name] = v
        eff[name] = hadamard(v, 512, signs[k], True)

    def aux(name, v):
        w[name] = eff[name] = np.asarray(v, np.float32)

    mat("model.embed_tokens.weight", 512, 512, (.05, .1))
    aux("model.norm.weight", np.zeros(512))
    for li, kind in enumerate(spec.kinds):
        p = f"model.layers.{li}."
        for n in ("input_layernorm", "post_attention_layernorm"):
            aux(p + n + ".weight", np.zeros(512))
        mat(p + "mlp.gate_proj.weight", 1024, 512)
        mat(p + "mlp.up_proj.weight", 1024, 512)
        mat(p + "mlp.down_proj.weight", 512, 1024)
        if kind == "linear":
            a = p + "linear_attn."
            mat(a + "in_proj_qkv.weight", 2048, 512)
            mat(a + "in_proj_z.weight", 1024, 512)
            mat(a + "out_proj.weight", 512, 1024)
            for k in ("in_proj_a.weight", "in_proj_b.weight"):
                aux(a + k, rng.normal(0, .02, (8, 512)))
            aux(a + "conv1d.weight", rng.normal(0, .2, (2048, 1, 4)))
            aux(a + "norm.weight", np.ones(128))
            aux(a + "A_log", np.zeros(8))
            aux(a + "dt_bias", np.full(8, -2.))
        else:
            a = p + "self_attn."
            for k, n, kk in [("q_proj", 2048, 512), ("k_proj", 512, 512),
                             ("v_proj", 512, 512), ("o_proj", 512, 1024)]:
                mat(a + k + ".weight", n, kk)
            aux(a + "q_norm.weight", np.zeros(256))
            aux(a + "k_norm.weight", np.zeros(256))
    H, d, F = spec.hidden, spec.head_dim, spec.ffn

    def lin(o, i):
        return (rng.uniform(-1, 1, (o, i)) / np.sqrt(i)).astype(np.float32)

    def nrm(n):
        return (0.1 * rng.standard_normal(n)).astype(np.float32)
    a = "mtp.layers.0.self_attn."
    for k, v in {"mtp.pre_fc_norm_embedding.weight": nrm(H),
                 "mtp.pre_fc_norm_hidden.weight": nrm(H),
                 "mtp.fc.weight": lin(H, 2 * H), "mtp.norm.weight": nrm(H),
                 "mtp.layers.0.input_layernorm.weight": nrm(H),
                 "mtp.layers.0.post_attention_layernorm.weight": nrm(H),
                 a + "q_proj.weight": lin(2 * spec.n_q * d, H),
                 a + "k_proj.weight": lin(spec.n_kv * d, H),
                 a + "v_proj.weight": lin(spec.n_kv * d, H),
                 a + "o_proj.weight": lin(H, spec.n_q * d),
                 a + "q_norm.weight": nrm(d), a + "k_norm.weight": nrm(d),
                 "mtp.layers.0.mlp.gate_proj.weight": lin(F, H),
                 "mtp.layers.0.mlp.up_proj.weight": lin(F, H),
                 "mtp.layers.0.mlp.down_proj.weight": lin(H, F)}.items():
        aux(k, v)
    eff["model.embed_tokens.weight"] = w.embedding_table()      # (the model's basis)
    return spec, w, eff


@pytest.fixture(autouse=True)
def _no_caches(monkeypatch):
    for key in ("OTPU_IMG_CACHE", "OTPU_IMAGE_CACHE", "OTPU_PROG_CACHE"):
        monkeypatch.setenv(key, "0")


@pytest.fixture(scope="module")
def tiny():
    return native_tiny()


PROMPT = [4, 9, 12, 7, 31, 5, 2, 40]
N = 12


@pytest.fixture(scope="module")
def plain(tiny):
    spec, w, _ = tiny
    return Engine(spec, w, resident=True, **KW).generate(PROMPT, max_new=N)


@pytest.mark.parametrize("drafter", ["mtp", "right", "wrong", "mixed"])
def test_bonsai_mtp_greedy_is_plain_greedy(tiny, plain, drafter):
    """The host loop: plain greedy's tokens whatever the drafts; the committed DeltaNet states
    and windows equal plain decode's word for word."""
    spec, w, _ = tiny
    want = plain
    assert len(set(want)) > 2, want                # (a model whose tokens vary)
    P = len(PROMPT)

    def right(q, out):
        return want[q - P] if q - P < len(want) else 0
    drafts = {"mtp": None, "right": right, "wrong": lambda q, out: (right(q, out) + 1) % 512,
              "mixed": lambda q, out: right(q, out) if q % 3 else (right(q, out) + 7) % 512}
    eng = mtp_engine(spec, w, **KW)
    dec = MTPDecoder(eng)
    st = dec.generate(PROMPT, max_new=N, drafts=drafts[drafter])
    assert st.tokens == want
    if drafter == "right":
        assert all(st.accepted[:-1]) and st.iterations <= N // 2 + 1
    if drafter == "wrong":
        assert not any(st.accepted)
    ref = Engine(spec, w, resident=True, **KW)
    ref.prefill(PROMPT)
    for t in want[:-1]:
        ref.step(t)
    assert eng.pos == ref.pos
    assert all(np.array_equal(a, b) for a, b in
               zip(_states(eng, spec, dec.slot), _states(ref, spec, 0)))


@pytest.mark.parametrize("drafter", ["right", "mixed"])
def test_bonsai_mtp_loop_on_the_device_is_plain_greedy(tiny, plain, drafter):
    """The loop on the device (V, E, D, D1 chained, docs/mtp.md 10): plain greedy's tokens."""
    spec, w, _ = tiny
    P = len(PROMPT)
    right = np.zeros(256, np.float32)
    right[P:P + len(plain)] = plain
    drafts = {"right": right, "mixed": np.where(np.arange(256) % 3, right, (right + 7) % 512)}
    eng = mtp_engine(spec, w, **KW)
    st = MTPDecoder(eng).generate_card(PROMPT, max_new=N, drafts=drafts[drafter])
    assert st.tokens == plain
    if drafter == "right":
        assert sum(st.accepted) >= len(plain) // 2 - 2


@pytest.mark.parametrize("fmt", ["int8", "fp4"])
def test_bonsai_mtp_program_follows_its_reference(tiny, fmt, monkeypatch):
    """The MTP program over 4 rows (the hidden rows from the rows kernel): its draft head's
    logits against mtp_reference on the fp32 model-basis weights' hidden rows (the MTP layer
    int8, or fp4 with OTPU_BONSAI_MTP_FORMAT; the draft head fp4), its draft ids their argmax."""
    monkeypatch.setenv("OTPU_BONSAI_MTP_FORMAT", fmt)
    spec, w, eff = tiny
    R = 4
    toks = PROMPT[:R + 1]
    eng = mtp_engine(spec, w, **KW)
    img = eng.image
    assert img.mtp_fc == fmt and all(f == fmt for f in img.keys[spec.layers][1])
    rs = replace(spec, native_bonsai=False)
    _, href = reference_logits(rs, eff, toks[:R], hidden=True)
    hid, got = [], []
    for j in range(R):      # (2-row I/O area: a row at a time, its MTP row after it)
        eng.backend.run(img.compile_rows([(0, j)], [], tokens=toks[j:j + 1], hidden=True))
        hid.append(eng.backend.read(0, img.io["hid"], 4 * spec.hidden).view(np.float32).copy())
        eng.backend.run(img.compile_mtp(j, 1, toks[j + 1:j + 2], keep=True))
        got.append(eng.backend.read(0, img.io["logits"], 4 * spec.vocab).view(np.float32)
                   [:img.nd].copy())
    hid, got = np.stack(hid), np.stack(got)
    cos = (hid * href).sum(1) / np.linalg.norm(hid, axis=1) / np.linalg.norm(href, axis=1)
    assert cos.min() > 0.99, cos
    want = mtp_reference(rs, eff, href, toks[1:R + 1])
    cos = (got * want).sum(1) / np.linalg.norm(got, axis=1) / np.linalg.norm(want, axis=1)
    assert cos.min() > (0.98 if fmt == "int8" else 0.95), cos
    if fmt == "int8":   # (fp4's random-weight logits are too flat for the argmax to hold)
        assert np.mean(np.argmax(got, 1) == np.argmax(want, 1)) >= 0.75


# ---- the GGUF's MTP layer (Ternary-Bonsai-2-4B-PQ2_0-MTP-Q8_0.gguf's layout, plain types)
def _write_gguf(path, kv, tensors):
    """A GGUF v3 file: kv {key: (type, value)}, tensors {name: (ggml type, shape [k, n], raw)}."""
    import struct

    def st(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    out = [b"GGUF", struct.pack("<IQQ", 3, len(tensors), len(kv))]
    for k, (t, v) in kv.items():
        out.append(st(k) + struct.pack("<I", t))
        out.append(st(v) if t == 8 else struct.pack({4: "<I", 6: "<f", 7: "<?"}[t], v))
    off, data = 0, []
    for n, (t, shape, raw) in tensors.items():
        out.append(st(n) + struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", d)
                                                                     for d in shape)
                   + struct.pack("<IQ", t, off))
        pad = (-len(raw)) % 32
        data.append(raw + b"\0" * pad)
        off += len(raw) + pad
    head = b"".join(out)
    head += b"\0" * ((-len(head)) % 32)
    path.write_bytes(head + b"".join(data))


def _q8_0(w):
    """llama.cpp Q8_0 of [n, k] rows (blocks of 32: f16 d, 32 int8)."""
    b = w.reshape(-1, 32)
    d = (np.abs(b).max(1) / 127).astype(np.float16)
    q = np.round(b / np.where(d == 0, 1, d.astype(np.float32))[:, None]).astype(np.int8)
    return (np.concatenate([d.view(np.uint8).reshape(-1, 2), q.view(np.uint8)], 1).tobytes(),
            (q.astype(np.float32) * d.astype(np.float32)[:, None]).reshape(w.shape))


def test_gguf_nextn_layer_maps_to_mtp(tmp_path):
    """GGUFWeights of an MTP GGUF: block_count counts the MTP layer (nextn_predict_layers 1),
    the Spec has the layers before it; blk.<L>.* reads as mtp.layers.0.*, nextn.eh_proj as
    mtp.fc (Q8_0, exact dequant), the nextn norms as mtp.* with llama.cpp's 1 + w undone."""
    from opentpu.llm.bonsai2 import GGUFWeights, spec_from_gguf
    rng = np.random.default_rng(5)
    H, F, d, nq, nkv, dk, nv, nk, V = 256, 512, 128, 2, 1, 128, 2, 2, 64
    kv = {"general.architecture": (8, "qwen35"), "qwen35.block_count": (4, 2),
          "qwen35.nextn_predict_layers": (4, 1), "qwen35.embedding_length": (4, H),
          "qwen35.feed_forward_length": (4, F), "qwen35.attention.head_count": (4, nq),
          "qwen35.attention.head_count_kv": (4, nkv), "qwen35.attention.key_length": (4, d),
          "qwen35.rope.dimension_count": (4, 64), "qwen35.ssm.time_step_rank": (4, nv),
          "qwen35.ssm.group_count": (4, nk), "qwen35.ssm.state_size": (4, dk),
          "qwen35.ssm.inner_size": (4, nv * 128), "qwen35.ssm.conv_kernel": (4, 4),
          "qwen35.attention.layer_norm_rms_epsilon": (6, 1e-6),
          "qwen35.rope.freq_base": (6, 1e7), "qwen35.full_attention_interval": (4, 4)}
    T, want = {}, {}

    def f32(name, shape, hf=None, plus=False):
        v = rng.standard_normal(shape[::-1]).astype(np.float32)
        T[name] = (0, list(shape), v.tobytes())
        if hf:
            want[hf] = v - 1 if plus else v

    def q8(name, k, n, hf):
        raw, v = _q8_0(rng.standard_normal((n, k)).astype(np.float32))
        T[name] = (8, [k, n], raw)
        want[hf] = v
    f32("token_embd.weight", [H, V])
    f32("output_norm.weight", [H])
    b = "blk.0."                     # one DeltaNet layer
    for n, s in [("attn_norm.weight", [H]), ("post_attention_norm.weight", [H]),
                 ("attn_qkv.weight", [H, 2 * nk * dk + nv * 128]),
                 ("attn_gate.weight", [H, nv * 128]), ("ssm_alpha.weight", [H, nv]),
                 ("ssm_beta.weight", [H, nv]), ("ssm_out.weight", [nv * 128, H]),
                 ("ssm_conv1d.weight", [4, 2 * nk * dk + nv * 128]), ("ssm_dt.bias", [nv]),
                 ("ssm_norm.weight", [128]), ("ffn_gate.weight", [H, F]),
                 ("ffn_up.weight", [H, F]), ("ffn_down.weight", [F, H])]:
        f32(b + n, s)
    T[b + "ssm_a"] = (0, [nv], (-np.ones(nv, np.float32)).tobytes())
    b = "blk.1."                     # the MTP layer (Q8_0 matrices, F32 norms)
    for n, hf in [("attn_norm.weight", "mtp.layers.0.input_layernorm.weight"),
                  ("post_attention_norm.weight", "mtp.layers.0.post_attention_layernorm.weight"),
                  ("nextn.enorm.weight", "mtp.pre_fc_norm_embedding.weight"),
                  ("nextn.hnorm.weight", "mtp.pre_fc_norm_hidden.weight"),
                  ("nextn.shared_head_norm.weight", "mtp.norm.weight")]:
        f32(b + n, [H], hf, plus=True)
    f32(b + "attn_q_norm.weight", [d], "mtp.layers.0.self_attn.q_norm.weight", plus=True)
    f32(b + "attn_k_norm.weight", [d], "mtp.layers.0.self_attn.k_norm.weight", plus=True)
    q8(b + "attn_q.weight", H, 2 * nq * d, "mtp.layers.0.self_attn.q_proj.weight")
    q8(b + "attn_k.weight", H, nkv * d, "mtp.layers.0.self_attn.k_proj.weight")
    q8(b + "attn_v.weight", H, nkv * d, "mtp.layers.0.self_attn.v_proj.weight")
    q8(b + "attn_output.weight", nq * d, H, "mtp.layers.0.self_attn.o_proj.weight")
    q8(b + "ffn_gate.weight", H, F, "mtp.layers.0.mlp.gate_proj.weight")
    q8(b + "ffn_up.weight", H, F, "mtp.layers.0.mlp.up_proj.weight")
    q8(b + "ffn_down.weight", F, H, "mtp.layers.0.mlp.down_proj.weight")
    q8(b + "nextn.eh_proj.weight", 2 * H, H, "mtp.fc.weight")
    p = tmp_path / "mtp.gguf"
    _write_gguf(p, kv, T)
    w = GGUFWeights(p)
    spec = spec_from_gguf(w)
    assert w.mtp and spec.layers == 1 and spec.kinds == ("linear",)
    for hf, v in want.items():
        np.testing.assert_array_equal(w[hf], v, err_msg=hf)
    assert not any(k.startswith("model.layers.1.") for k in w)


def test_bonsai_mtp_weights_pair_the_pack_and_the_gguf(tmp_path, monkeypatch):
    """BonsaiMTPWeights: mtp.* from the GGUF, the rest from the pack; a pack whose compared
    projections differ from the GGUF's model is refused; find_mtp_gguf's places."""
    from opentpu.llm import bonsai2 as B
    test_gguf_nextn_layer_maps_to_mtp(tmp_path)
    g = B.GGUFWeights(tmp_path / "mtp.gguf")
    base = {n: g[n] for n in g if not n.startswith("mtp.")}
    w = B.BonsaiMTPWeights(base, g)
    assert set(w) == set(g)
    np.testing.assert_array_equal(w["mtp.fc.weight"], g["mtp.fc.weight"])
    bad = dict(base)
    bad["model.layers.0.mlp.down_proj.weight"] = bad["model.layers.0.mlp.down_proj.weight"] * 2
    with pytest.raises(ValueError, match="differs from the pack"):
        B.BonsaiMTPWeights(bad, g)
    pack = tmp_path / "models" / "pack"
    (tmp_path / "models" / "bonsai2-4b-mtp").mkdir(parents=True)
    pack.mkdir()
    with pytest.raises(ValueError, match="no \\*MTP\\*.gguf"):
        B.find_mtp_gguf(pack)
    f = tmp_path / "models" / "bonsai2-4b-mtp" / "X-PQ2_0-MTP-Q8_0.gguf"
    f.write_bytes(b"")
    assert B.find_mtp_gguf(pack) == f
    monkeypatch.setenv("OTPU_BONSAI_MTP", "/elsewhere.gguf")
    assert str(B.find_mtp_gguf(pack)) == "/elsewhere.gguf"


def test_image_cache_identity_of_a_gguf(tmp_path):
    """imgcache.checkpoint of a single file (the MTP GGUF): its name, size, time and header hash;
    BonsaiMTPWeights carries it as mtp_from, the pack's directory as model_dir."""
    from opentpu import imgcache as IC
    from opentpu.llm import bonsai2 as B
    test_gguf_nextn_layer_maps_to_mtp(tmp_path)
    f = tmp_path / "mtp.gguf"
    a = IC.checkpoint(f)
    assert len(a) == 1 and a[0][0] == "mtp.gguf" and a[0][1] == f.stat().st_size
    g = B.GGUFWeights(f)
    base = {n: g[n] for n in g if not n.startswith("mtp.")}
    w = B.BonsaiMTPWeights(base, g)
    assert w.mtp_from == f and w.model_dir is None


def test_bonsai_mtp_verify_uses_grq_quad_of_two_rows(tiny, plain):
    """MCOLS 8 with GRQ QUAD (TGQ4) and its 2-row form (TGQ4M): the verify's ternary projections
    are GRQ QUAD TMMs of M 2 (a quarter of plain GRQ's advances, docs/isa.md), and the greedy
    tokens are still plain greedy decode's."""
    from opentpu import isa as I
    spec, w, _ = tiny
    cfg = replace(CFG, MCOLS=8, TGQ4=True, TGQ4M=True)
    kw = dict(KW, cfg=cfg)
    want = Engine(spec, w, resident=True, **kw).generate(PROMPT, max_new=N)
    eng = mtp_engine(spec, w, **kw)
    prog = eng.image.compile_rows([(0, 6), (0, 7)], [0, 1], tokens=[4, 5], fork=True,
                                  hidden=True)
    words = np.asarray(I.assemble(prog[0]) if not isinstance(prog[0], np.ndarray) else prog[0],
                       "<u4")
    op, dims, w6 = words[0::8], words[4::8], words[6::8]
    tmm = ((op & 0xff) == I.TMM) & ((dims & I.TMM_GRQ) != 0)
    quad2 = tmm & (((op >> 24) & I.F_QUAD) != 0) & (((w6 >> 16) & 0xff) == 2)
    assert quad2.sum() > 0 and quad2.sum() == (tmm & (((w6 >> 16) & 0xff) == 2)).sum()
    st = MTPDecoder(eng).generate(PROMPT, max_new=N, drafts=lambda q, out: (
        want[q - len(PROMPT)] if q - len(PROMPT) < len(want) else 0))
    assert st.tokens == want
