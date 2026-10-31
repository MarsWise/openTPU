"""Independent checks of Prism transforms and native 2-bit/group-128 arithmetic."""
import numpy as np
import pytest

from opentpu.llm.bonsai2 import NativeMatrix, hadamard, unpack_ternary


def sylvester(n):
    h = np.ones((1, 1))
    while len(h) < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def matrix(n=5, width=1024):
    rng = np.random.default_rng(121)
    trits = rng.integers(-1, 2, (n, width), dtype=np.int8)
    codes = trits.astype(np.uint32) + np.uint32(1)
    packed = np.zeros((n, width // 16), np.uint32)
    for k in range(16):
        packed |= codes[:, k::16] << np.uint32(2 * k)
    scales = rng.uniform(.01, .1, (n, width // 128)).astype(np.float16)
    signs = rng.choice(np.array([-1, 1], np.float32), width)
    return NativeMatrix(packed, scales, -scales, signs, embedding=True), trits


def test_hadamard_against_independent_dense_matrix():
    rng = np.random.default_rng(77)
    x = rng.normal(size=(3, 1024)).astype(np.float32)
    signs = rng.choice(np.array([-1, 1], np.float32), 1024)
    expected = ((x * signs).reshape(-1, 512) @ sylvester(512)).reshape(x.shape)
    np.testing.assert_allclose(hadamard(x, 512, signs), expected, atol=8e-7, rtol=1e-5)
    np.testing.assert_allclose(hadamard(hadamard(x, 512, signs), 512, signs, True),
                               x, atol=8e-7, rtol=1e-5)


def test_native_codes_scales_and_tied_embedding():
    mat, trits = matrix()
    actual, scales = mat.unpack()
    np.testing.assert_array_equal(actual, trits)
    np.testing.assert_array_equal(scales, mat.scales.astype(np.float32))
    from opentpu.quant import unpack_ternary as unpack_tmm
    packed, scales = mat.tmm()
    np.testing.assert_array_equal(unpack_tmm(packed, 1024), trits)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(3, 1024)).astype(np.float32)
    dense = (trits.reshape(5, -1, 128) * scales[..., None]).reshape(5, 1024)
    h = sylvester(512)
    effective = ((dense.reshape(-1, 512) @ h).reshape(dense.shape) * mat.signs)
    np.testing.assert_allclose(mat.forward_reference(x), x @ effective.T, atol=2e-6, rtol=1e-5)
    np.testing.assert_allclose(mat.embedding_reference([0, 2]), effective[[0, 2]], atol=8e-8)


def test_refuse_affine_weights_outside_ternary_contract():
    mat, _ = matrix()
    bad = mat.weight.copy()
    bad[0, 0] |= np.uint32(3)
    with pytest.raises(ValueError, match="Reserved"):
        unpack_ternary(bad, mat.scales, mat.biases)
    with pytest.raises(ValueError, match="symmetric"):
        unpack_ternary(mat.weight, mat.scales, np.zeros_like(mat.biases))


@pytest.mark.parametrize("inverse", [False, True])
def test_vpu_hadamard_kernel(inverse):
    from opentpu import language as ol
    from opentpu.isasim import Config
    from opentpu.kernels.hadamard import hadamard as kernel_hadamard
    from opentpu.runtime import Input, Output, compile_kernel, run
    rng = np.random.default_rng(19)
    x = rng.normal(size=(2, 1024)).astype(np.float32)
    signs = rng.choice(np.array([-1, 1], np.float32), 1024)

    @ol.jit
    def transform(src, sg, dst):
        y = kernel_hadamard(ol.load(src), ol.load(sg), inverse=inverse)
        ol.store(dst, y)

    compiled, images = compile_kernel(transform, Config(D=128), src=Input(x),
                                      sg=Input(signs), dst=Output(x.shape))
    result = run(compiled, images, "isa").outputs["dst"]
    np.testing.assert_allclose(result, hadamard(x, 512, signs, inverse), atol=8e-7, rtol=1e-5)


@pytest.mark.parametrize("grq", [False, True])
@pytest.mark.parametrize("dstep", [False, True])
def test_native_hybrid_decoder_against_unfolded_float_reference(monkeypatch, dstep, grq):
    """Both layer kinds and the tied head; the reference unfolds weights, the device rotates x."""
    from dataclasses import replace
    from opentpu.isasim import Config
    from opentpu.llm.qwen35 import Spec, reference_logits
    from opentpu.llm.qwen3 import Engine
    for key in ("OTPU_IMG_CACHE", "OTPU_IMAGE_CACHE", "OTPU_PROG_CACHE"):
        monkeypatch.setenv(key, "0")
    spec = Spec(512, ("linear", "attn"), 4, 2, 256, 64, 8, 128, 128, 1024, 512,
                lin_kheads=4, native_bonsai=True, pair_loop=True)
    rng = np.random.default_rng(333)
    signs = {k: rng.choice(np.array([-1, 1], np.float32), k) for k in (512, 1024)}

    class W(dict):
        def signs_for(self, width):
            return signs[width]

        def embedding_table(self):
            return hadamard(self['model.embed_tokens.weight'], 512, signs[512], True)

    w = W()
    effective = {}
    def matrix_(name, n, k):
        groups = rng.uniform(.01, .04, (n, k//128)).astype(np.float16).astype(np.float32)
        value = (rng.integers(-1, 2, (n, k), dtype=np.int8).reshape(n, -1, 128)
                 * groups[..., None]).reshape(n, k)
        w[name] = value
        effective[name] = hadamard(value, 512, signs[k], True)

    def aux(name, value):
        w[name] = effective[name] = np.asarray(value, np.float32)

    matrix_('model.embed_tokens.weight', 512, 512)
    aux('model.norm.weight', np.zeros(512))
    for li, kind in enumerate(spec.kinds):
        p = f'model.layers.{li}.'
        for norm in ('input_layernorm', 'post_attention_layernorm'):
            aux(p+norm+'.weight', np.zeros(512))
        matrix_(p+'mlp.gate_proj.weight', 1024, 512)
        matrix_(p+'mlp.up_proj.weight', 1024, 512)
        matrix_(p+'mlp.down_proj.weight', 512, 1024)
        if kind == 'linear':
            a = p+'linear_attn.'
            matrix_(a+'in_proj_qkv.weight', 2048, 512)
            matrix_(a+'in_proj_z.weight', 1024, 512)
            matrix_(a+'out_proj.weight', 512, 1024)
            for key in ('in_proj_a.weight', 'in_proj_b.weight'):
                aux(a+key, rng.normal(0, .02, (8, 512)))
            aux(a+'conv1d.weight', rng.normal(0, .2, (2048, 1, 4)))
            aux(a+'norm.weight', np.ones(128))
            aux(a+'A_log', np.zeros(8))
            aux(a+'dt_bias', np.full(8, -2.))
        else:
            a = p+'self_attn.'
            for key, n, k in [('q_proj', 2048, 512), ('k_proj', 512, 512),
                              ('v_proj', 512, 512), ('o_proj', 512, 1024)]:
                matrix_(a+key+'.weight', n, k)
            aux(a+'q_norm.weight', np.zeros(256))
            aux(a+'k_norm.weight', np.zeros(256))
    cfg = Config(D=128, MCOLS=4, ACT_ROWS=8, TGSC=True, DRAM_BYTES=32 << 20,
                 DSTEP=dstep, STREAM=dstep, TFN=dstep, PAIR=True, TQUAD=True, TPAIR=True,
                 TGRQ=grq)
    eng = Engine(spec, w, cap=128, cfg=cfg, resident=True, pipeline=False,
                 rows=1, wformat='ternary_gsc')
    tokens = [4, 9, 12, 7, 31]
    ref = reference_logits(replace(spec, native_bonsai=False), effective, tokens)
    actual = np.stack([eng.step(t) for t in tokens])
    cosine = (actual * ref).sum(1) / np.linalg.norm(actual, axis=1) / np.linalg.norm(ref, axis=1)
    assert cosine.min() > .99
    assert all(i.op != 0x20 or i.flags >> 4 & 3 == 0 for p in eng.image.compile_step(0, tok=4)
               for i in p)   # no FP4 projection fallback
    from opentpu import isa as I
    tmms = [i for p in eng.image.compile_step(0, tok=4) for i in p if i.op == I.TMM]
    assert any(i.w[3] & I.TMM_GRQ for i in tmms) == grq
    if grq:     # the block-major layout is the serial GSC path's math, bit for bit
        row = Engine(spec, w, cap=128, cfg=replace(cfg, TGRQ=False), resident=True,
                     pipeline=False, rows=1, wformat='ternary_gsc')
        np.testing.assert_array_equal(actual, np.stack([row.step(t) for t in tokens]))


def test_native_hadamard_gsc_linear_rtl(have_verilator):
    from opentpu import language as ol, rtlsim
    from opentpu.compiler import QTensor
    from opentpu.isasim import Config
    from opentpu.runtime import Input, Output, compile_kernel, run
    mat, _ = matrix(n=7)
    rng = np.random.default_rng(911)
    x = rng.normal(size=(2, 1024)).astype(np.float32)
    cfg = Config(D=128, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=16, TGSC=True)
    weight = QTensor.weight(0x40000, 0x60000, 7, 1024, 'ternary_gsc', 128)

    @ol.jit
    def linear(src, dst, w):
        ol.store(dst, ol.dot(ol.load(src), w))

    compiled, images = compile_kernel(linear, cfg, src=Input(x), dst=Output((2, 7)), w=weight)
    packed, scales = mat.tmm()
    for addr, data in ((0x40000, packed), (0x60000, scales),
                       (0x60000 + scales.nbytes, mat.signs)):
        raw = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        images[0][addr:addr+len(raw)] = raw
    ref = run(compiled, [images[0].copy()], 'isa')
    actual = ref.outputs['dst']
    want = mat.forward_reference(x)
    assert np.linalg.norm(actual-want)/np.linalg.norm(want) < .012
    drams, tmems, _ = rtlsim.run(cfg, compiled.programs, images, axi=True, stall=35, seed=891)
    np.testing.assert_array_equal(drams[0], ref.drams[0])
    np.testing.assert_array_equal(tmems[0], ref.tmems[0])
