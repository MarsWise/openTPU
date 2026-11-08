"""CACT CQ2/CQ4 format and independent rotation/matmul reference checks."""
import ast
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from opentpu.cactus.cq import read_cact


def fixture(path, bits, inter):
    rng = np.random.default_rng(192 + bits + inter)
    n, gs, groups = 8 if inter else 5, 128, 2
    k = gs * groups
    idx = rng.integers(0, 1 << bits, (n, k), dtype=np.uint8)
    cb = rng.uniform(-1, 1, 1 << bits).astype('<f2')
    norms = rng.uniform(.1, 2, (n, groups)).astype('<f2')
    scale = rng.uniform(.5, 2, k).astype('<f2')
    recip = (1 / scale.astype(np.float32)).astype('<f2')
    left = rng.choice(np.array([-1, 1], np.int8), gs)
    right = rng.choice(np.array([-1, 1], np.int8), gs)
    perm = rng.permutation(gs).astype('<u4')
    if inter:
        b = bytearray()
        for nb in range(n // 4):
            for g in range(groups):
                if bits == 4:
                    for chunk in range(gs // 8):
                        for row in range(4):
                            a = idx[4 * nb + row, g * gs + chunk * 8:g * gs + chunk * 8 + 8]
                            b.extend(int(a[i]) | int(a[i + 4]) << 4 for i in range(4))
                else:
                    for chunk in range(gs // 16):
                        for p in range(4):
                            for row in range(4):
                                a = idx[4 * nb + row, g * gs + chunk * 16 + p * 4:g * gs + chunk * 16 + p * 4 + 4]
                                b.append(sum(int(a[i]) << (2 * i) for i in range(4)))
        normblob = norms.reshape(n // 4, 4, groups).transpose(0, 2, 1).copy().tobytes()
    else:
        b = bytearray()
        for a in idx.reshape(-1, 8 // bits):
            b.append(sum(int(v) << (bits * i) for i, v in enumerate(a)))
        normblob = norms.tobytes()
    meta = b''.join((cb.tobytes(), scale.tobytes(), recip.tobytes(), normblob,
                     left.tobytes(), right.tobytes(), perm.tobytes()))
    header = struct.pack('<4sIII4QIQQIIQ', b'CACT', 4 if inter else 0, 32, 2,
                         n, k, 0, 0, 4 if bits == 2 else 6, len(b), len(meta), gs, groups, n)
    data = header + bytes((-len(header)) % 32) + meta
    data += bytes((-len(data)) % 32) + b
    path.write_bytes(data)
    return idx, cb, norms, scale, recip, left, right, perm


@pytest.mark.parametrize('bits', [2, 4])
@pytest.mark.parametrize('inter', [False, True])
def test_cact_layout_and_independent_matrix_reference(tmp_path, bits, inter):
    path = tmp_path / 'weights.cact'
    idx, cb, norms, scale, recip, left, right, perm = fixture(path, bits, inter)
    cq = read_cact(path)
    np.testing.assert_array_equal(cq.indices, idx)
    np.testing.assert_array_equal(cq.norms, norms.astype(np.float32))
    np.testing.assert_array_equal(cq.input_scale, scale.astype(np.float32))
    h = np.ones((1, 1), np.float32)
    while h.shape[0] < 128:
        h = np.block([[h, h], [h, -h]])
    rot = (left[:, None] * h / np.sqrt(128) * right[None, :])[:, perm]
    rng = np.random.default_rng(91)
    x = rng.normal(size=(3, idx.shape[1])).astype(np.float32)
    basis = (x * recip.astype(np.float32)).reshape(3, 2, 128) @ rot
    ref = np.zeros((3, len(idx)), np.float32)
    for j in range(len(idx)):
        for g in range(2):
            ref[:, j] += (basis[:, g] @ cb[idx[j, g * 128:(g + 1) * 128]].astype(np.float32)) * norms[j, g]
    np.testing.assert_allclose(cq.matmul_reference(x), ref, rtol=1e-5, atol=1e-5)
    folded = cq.fold_permutation_into_indices()
    assert np.array_equal(folded.permutation, np.arange(128))
    np.testing.assert_allclose(folded.matmul_reference(x), cq.matmul_reference(x), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(cq.indices, idx)  # offline conversion must not mutate the source


def test_cact_rejects_truncated_unsupported_and_invalid_rotation(tmp_path):
    p = tmp_path / 'bad.cact'
    fixture(p, 2, False)
    raw = p.read_bytes()
    p.write_bytes(raw[:-1])
    with pytest.raises(ValueError, match='length'):
        read_cact(p)
    damaged = bytearray(raw)
    struct.pack_into('<I', damaged, 4, 2)  # orthogonal, not Hadamard
    p.write_bytes(damaged)
    with pytest.raises(ValueError, match='orthogonal'):
        read_cact(p)
    damaged = bytearray(raw)
    struct.pack_into('<I', damaged, 48, 5)  # CQ3, unsupported
    p.write_bytes(damaged)
    with pytest.raises(ValueError, match='precision'):
        read_cact(p)


@pytest.mark.parametrize('groups,identity', [(1, True), (2, False)])
def test_cq_vpu_activation_matches_cact_reference(tmp_path, groups, identity):
    from opentpu import language as ol
    from opentpu.isasim import Config
    from opentpu.kernels.cactus_cq import cq_activation
    from opentpu.runtime import Input, Output, compile_kernel, run

    path = tmp_path / 'weights.cact'
    fixture(path, 4, False)
    cq = read_cact(path)
    from dataclasses import replace
    width = groups * cq.group_size
    perm = np.arange(cq.group_size) if identity else cq.permutation
    cq = replace(cq, indices=cq.indices[:, :width], norms=cq.norms[:, :groups],
                 input_scale=cq.input_scale[:width], input_scale_recip=cq.input_scale_recip[:width],
                 permutation=perm)
    rng = np.random.default_rng(918)
    x = rng.normal(size=(2, width)).astype(np.float32)
    left = np.tile(cq.left_signs, groups).astype(np.float32)
    right = np.tile(cq.right_signs, groups).astype(np.float32)
    permutation = tuple(map(int, perm))

    @ol.jit
    def transform(src, reciprocal, lsign, rsign, dst):
        y = cq_activation(ol.load(src), ol.load(reciprocal), ol.load(lsign),
                          ol.load(rsign), permutation)
        ol.store(dst, y)

    compiled, images = compile_kernel(transform, Config(D=128), src=Input(x),
                                      reciprocal=Input(cq.input_scale_recip), lsign=Input(left),
                                      rsign=Input(right), dst=Output(x.shape))
    result = run(compiled, images, 'isa').outputs['dst']
    np.testing.assert_allclose(result, cq.transform(x).reshape(x.shape), rtol=1e-5, atol=2e-6)


def test_cq_vpu_activation_matches_isa_rtl(have_verilator):
    from opentpu import language as ol, rtlsim
    from opentpu.isasim import Config
    from opentpu.kernels.cactus_cq import cq_activation
    from opentpu.runtime import Input, Output, collect, compile_kernel, run

    rng = np.random.default_rng(582)
    x = rng.normal(size=(2, 128)).astype(np.float32)
    recip = rng.uniform(.5, 2, size=128).astype(np.float32)
    left = rng.choice([-1., 1.], size=128).astype(np.float32)
    right = rng.choice([-1., 1.], size=128).astype(np.float32)
    permutation = tuple(map(int, rng.permutation(128)))

    @ol.jit
    def transform(src, reciprocal, lsign, rsign, dst):
        ol.store(dst, cq_activation(ol.load(src), ol.load(reciprocal), ol.load(lsign),
                                    ol.load(rsign), permutation))

    cfg = Config(D=128, MCOLS=4, DRAM_BYTES=1 << 20)
    compiled, images = compile_kernel(transform, cfg, src=Input(x), reciprocal=Input(recip),
                                      lsign=Input(left), rsign=Input(right), dst=Output(x.shape))
    expected = run(compiled, [a.copy() for a in images], 'isa').outputs['dst']
    drams, tmems, stats = rtlsim.run(cfg, compiled.programs, images, axi=True, stall=35, seed=55)
    actual = collect(compiled, drams, tmems, stats).outputs['dst']
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('bits,inter', [(2, False), (4, True)])
def test_cq_offline_int8_baseline_executes_vpu_and_mxu(tmp_path, bits, inter):
    from opentpu import language as ol
    from opentpu.cactus.transcode import cq_to_mxu_int8
    from opentpu.compiler import QTensor
    from opentpu.isasim import Config
    from opentpu.kernels.cactus_cq import cq_activation
    from opentpu.runtime import Input, Output, compile_kernel, run

    path = tmp_path / 'weights.cact'
    fixture(path, bits, inter)
    cq = read_cact(path)
    folded, q, scales = cq_to_mxu_int8(cq, chunk_rows=3)
    _, q2, scales2 = cq_to_mxu_int8(cq, chunk_rows=8)
    np.testing.assert_array_equal(q, q2)
    np.testing.assert_array_equal(scales, scales2)
    n, k = q.shape
    rng = np.random.default_rng(552)
    x = rng.normal(size=(2, k)).astype(np.float32)
    left = np.tile(folded.left_signs, k // 128).astype(np.float32)
    right = np.tile(folded.right_signs, k // 128).astype(np.float32)
    perm = tuple(map(int, folded.permutation))
    weight = QTensor.weight(0x20000, 0x30000, n, k, 'int8', 128)

    @ol.jit
    def projection(src, reciprocal, lsign, rsign, dst, weight):
        basis = cq_activation(ol.load(src), ol.load(reciprocal), ol.load(lsign), ol.load(rsign), perm)
        ol.store(dst, ol.dot(basis, weight))

    cfg = Config(D=128, MCOLS=4, DRAM_BYTES=1 << 20)
    compiled, images = compile_kernel(projection, cfg, src=Input(x),
                                      reciprocal=Input(folded.input_scale_recip),
                                      lsign=Input(left), rsign=Input(right), dst=Output((2, n)),
                                      weight=weight)
    for address, values in ((0x20000, q), (0x30000, scales)):
        raw = np.ascontiguousarray(values).view(np.uint8).reshape(-1)
        images[0][address:address + len(raw)] = raw
    actual = run(compiled, images, 'isa').outputs['dst']
    expected = cq.matmul_reference(x)
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < .05


def test_cq_offline_int8_projection_matches_isa_rtl(tmp_path, have_verilator):
    from dataclasses import replace
    from opentpu import language as ol, rtlsim
    from opentpu.cactus.transcode import cq_to_mxu_int8
    from opentpu.compiler import QTensor
    from opentpu.isasim import Config
    from opentpu.kernels.cactus_cq import cq_activation
    from opentpu.runtime import Input, Output, collect, compile_kernel, run

    path = tmp_path / 'weights.cact'
    fixture(path, 2, False)
    cq = read_cact(path)
    cq = replace(cq, indices=cq.indices[:, :128], norms=cq.norms[:, :1],
                 input_scale=cq.input_scale[:128], input_scale_recip=cq.input_scale_recip[:128])
    folded, q, scales = cq_to_mxu_int8(cq)
    rng = np.random.default_rng(892)
    x = rng.normal(size=(2, 128)).astype(np.float32)
    w = QTensor.weight(0x20000, 0x30000, len(q), 128, 'int8', 128)

    @ol.jit
    def projection(src, reciprocal, left, right, dst, w):
        basis = cq_activation(ol.load(src), ol.load(reciprocal), ol.load(left), ol.load(right),
                              tuple(range(128)))
        ol.store(dst, ol.dot(basis, w))

    cfg = Config(D=128, MCOLS=4, DRAM_BYTES=1 << 20)
    compiled, images = compile_kernel(projection, cfg, src=Input(x),
                                      reciprocal=Input(folded.input_scale_recip),
                                      left=Input(folded.left_signs.astype(np.float32)),
                                      right=Input(folded.right_signs.astype(np.float32)),
                                      dst=Output((2, len(q))), w=w)
    for address, values in ((0x20000, q), (0x30000, scales)):
        raw = np.ascontiguousarray(values).view(np.uint8).reshape(-1)
        images[0][address:address + len(raw)] = raw
    expected = run(compiled, [a.copy() for a in images], 'isa').outputs['dst']
    drams, tmems, stats = rtlsim.run(cfg, compiled.programs, images, axi=True, stall=35, seed=56)
    actual = collect(compiled, drams, tmems, stats).outputs['dst']
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('bits', [2, 4])
@pytest.mark.parametrize('inter', [False, True])
def test_layout_against_cactus_upstream_packer(tmp_path, bits, inter):
    source = os.environ.get('CACTUS_SOURCE')
    if not source:
        pytest.skip('set CACTUS_SOURCE to the audited cactus checkout')
    code = (Path(source) / 'python/cactus/convert/quantization/cq.py').read_text(encoding='utf-8')
    tree = ast.parse(code)
    name = f'pack_indices_interleaved_4row_{bits}bit' if inter else 'pack_indices_lsb'
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
    ns = {'np': np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), ns)
    path = tmp_path / 'test.cact'
    expected = fixture(path, bits, inter)[0]
    packed = ns[name](expected, 128) if inter else ns[name](expected, 128, bits)
    raw = path.read_bytes()
    assert raw[-packed.size:] == packed.tobytes()
    np.testing.assert_array_equal(read_cact(path).indices, expected)
