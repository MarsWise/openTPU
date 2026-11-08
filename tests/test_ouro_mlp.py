"""Ouro sandwich MLP primitive: CPU reference, ISA and optional RTL."""
import numpy as np
import pytest


def test_attention_qkv_isa_matches_int4_reference():
    from opentpu import language as ol
    from opentpu.compiler import QTensor
    from opentpu.isasim import Config
    from opentpu.kernels.ouro import attention_qkv
    from opentpu.quant import quantize_mxu, dequantize_mxu
    from opentpu.runtime import Input, Output, compile_kernel, run

    rng = np.random.default_rng(813)
    x = rng.normal(size=(2, 128)).astype(np.float32)
    gamma = rng.uniform(.8, 1.2, 128).astype(np.float32)
    mats = [rng.normal(0, .05, (128, 128)).astype(np.float32) for _ in range(3)]
    offsets = [(0x20000, 0x30000), (0x40000, 0x50000), (0x60000, 0x70000)]
    weights = [QTensor.weight(d, s, 128, 128, 'int4', 128) for d, s in offsets]

    @ol.jit
    def kernel(src, g, qout, kout, vout, wq, wk, wv):
        q, k, v = attention_qkv(ol.load(src), ol.load(g), wq, wk, wv, 1e-6)
        ol.store(qout, q)
        ol.store(kout, k)
        ol.store(vout, v)

    cfg = Config(D=128, MCOLS=4, DRAM_BYTES=1 << 20)
    compiled, images = compile_kernel(kernel, cfg, src=Input(x), g=Input(gamma),
                                      qout=Output(x.shape), kout=Output(x.shape),
                                      vout=Output(x.shape), wq=weights[0], wk=weights[1], wv=weights[2])
    expected = []
    normalized = x / np.sqrt(np.mean(x*x, axis=1, keepdims=True) + 1e-6) * gamma
    for (d, s), matrix in zip(offsets, mats):
        packed, scales = quantize_mxu(matrix, 'int4', 128)
        for addr, arr in ((d, packed), (s, scales)):
            raw = np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
            images[0][addr:addr + len(raw)] = raw
        expected.append(normalized @ dequantize_mxu(packed, scales, 'int4', 128).T)
    result = run(compiled, images, 'isa').outputs
    for key, reference in zip(('qout', 'kout', 'vout'), expected):
        assert np.linalg.norm(result[key] - reference) / np.linalg.norm(reference) < .05


def case():
    from opentpu import language as ol
    from opentpu.compiler import QTensor
    from opentpu.isasim import Config
    from opentpu.kernels.ouro import sandwich_mlp
    from opentpu.quant import quantize_mxu, dequantize_mxu
    from opentpu.runtime import Input, Output, compile_kernel

    rng = np.random.default_rng(1003)
    hidden, ffn = 128, 256
    x = rng.normal(size=(2, hidden)).astype(np.float32)
    gb = rng.uniform(.8, 1.2, hidden).astype(np.float32)
    ga = rng.uniform(.8, 1.2, hidden).astype(np.float32)
    mats = [rng.normal(0, .06, size=shape).astype(np.float32)
            for shape in ((ffn, hidden), (ffn, hidden), (hidden, ffn))]
    loc = [(0x20000, 0x30000), (0x40000, 0x50000), (0x60000, 0x70000)]
    tensors = [QTensor.weight(d, s, *w.shape, 'int4', 128) for (d, s), w in zip(loc, mats)]

    @ol.jit
    def kernel(src, before, after, dst, gate, up, down):
        ol.store(dst, sandwich_mlp(ol.load(src), ol.load(before), ol.load(after),
                                   gate, up, down, 1e-6))

    cfg = Config(D=128, MCOLS=4, DRAM_BYTES=1 << 20)
    compiled, images = compile_kernel(kernel, cfg, src=Input(x), before=Input(gb),
                                      after=Input(ga), dst=Output(x.shape),
                                      gate=tensors[0], up=tensors[1], down=tensors[2])
    deq = []
    for (d, s), mat in zip(loc, mats):
        q, scales = quantize_mxu(mat, 'int4', 128)
        deq.append(dequantize_mxu(q, scales, 'int4', mat.shape[1]))
        for address, values in ((d, q), (s, scales)):
            raw = np.ascontiguousarray(values).view(np.uint8).reshape(-1)
            images[0][address:address + len(raw)] = raw
    def norm(v, g):
        return v / np.sqrt(np.mean(v * v, axis=-1, keepdims=True) + 1e-6) * g
    xn = norm(x, gb)
    g, u = xn @ deq[0].T, xn @ deq[1].T
    mlp = ((g / (1 + np.exp(-g))) * u) @ deq[2].T
    ref = x + norm(mlp, ga)
    return cfg, compiled, images, ref, (x, gb, ga, deq)


def test_sandwich_mlp_isa_matches_quantized_reference():
    from opentpu.runtime import run
    _, compiled, images, reference, _ = case()
    actual = run(compiled, images, 'isa').outputs['dst']
    assert np.linalg.norm(actual - reference) / np.linalg.norm(reference) < .05


def test_sandwich_mlp_rtl_matches_isa(have_verilator):
    from opentpu import rtlsim
    from opentpu.runtime import collect, run
    cfg, compiled, images, _, _ = case()
    expected = run(compiled, [a.copy() for a in images], 'isa').outputs['dst']
    drams, tmems, stats = rtlsim.run(cfg, compiled.programs, images, axi=True, stall=35, seed=81)
    actual = collect(compiled, drams, tmems, stats).outputs['dst']
    np.testing.assert_array_equal(actual, expected)


def test_sandwich_mlp_isa_against_pytorch():
    torch = pytest.importorskip('torch')
    from opentpu.runtime import run
    _, compiled, images, _, (x, gb, ga, matrices) = case()
    tx, tgb, tga = (torch.from_numpy(a) for a in (x, gb, ga))
    gate, up, down = (torch.from_numpy(a).float() for a in matrices)
    def norm(t, gamma):
        return t * torch.rsqrt(torch.mean(t * t, dim=-1, keepdim=True) + 1e-6) * gamma
    xn = norm(tx, tgb)
    mlp = (torch.nn.functional.silu(xn @ gate.T) * (xn @ up.T)) @ down.T
    expected = (tx + norm(mlp, tga)).numpy()
    actual = run(compiled, images, 'isa').outputs['dst']
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < .05


def test_attention_norms_isa_rtl_and_pytorch(have_verilator):
    torch = pytest.importorskip('torch')
    from opentpu import language as ol, rtlsim
    from opentpu.isasim import Config
    from opentpu.kernels.ouro import attention_input, attention_residual
    from opentpu.runtime import Input, Output, collect, compile_kernel, run

    rng = np.random.default_rng(28)
    x, attn = (rng.normal(size=(2, 128)).astype(np.float32) for _ in range(2))
    gb, ga = (rng.uniform(.8, 1.2, size=128).astype(np.float32) for _ in range(2))

    @ol.jit
    def kernel(src, branch, before, after, normed, result):
        original = ol.load(src)
        ol.store(normed, attention_input(original, ol.load(before), 1e-6))
        ol.store(result, attention_residual(original, ol.load(branch), ol.load(after), 1e-6))

    cfg = Config(D=128, MCOLS=4, DRAM_BYTES=1 << 20)
    compiled, images = compile_kernel(kernel, cfg, src=Input(x), branch=Input(attn),
                                      before=Input(gb), after=Input(ga),
                                      normed=Output(x.shape), result=Output(x.shape))
    expected = run(compiled, [a.copy() for a in images], 'isa').outputs
    tx, ta, tgb, tga = (torch.from_numpy(a) for a in (x, attn, gb, ga))
    def norm(t, gamma):
        return t * torch.rsqrt(torch.mean(t * t, dim=-1, keepdim=True) + 1e-6) * gamma
    np.testing.assert_allclose(expected['normed'], norm(tx, tgb).numpy(), atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(expected['result'], (tx + norm(ta, tga)).numpy(), atol=2e-5, rtol=2e-5)
    drams, tmems, stats = rtlsim.run(cfg, compiled.programs, images, axi=True, stall=35, seed=82)
    actual = collect(compiled, drams, tmems, stats).outputs
    for name in ('normed', 'result'):
        np.testing.assert_array_equal(actual[name], expected[name])
