"""CACT tensor to existing TMM CQ + VPU path, synthetic data only."""
from dataclasses import replace

import numpy as np
import pytest

from opentpu import isa as I
from opentpu.cactus.cq import read_cact
from test_cactus_cq import fixture


@pytest.mark.parametrize("bits,inter", [(2, False), (2, True), (4, False), (4, True)])
def test_cact_native_tmm_cq_with_device_basis(tmp_path, bits, inter):
    from opentpu import cq as device_cq, language as ol
    from opentpu.cactus.transcode import cq_to_tmm_cq
    from opentpu.compiler import CompileError, QTensor
    from opentpu.isasim import Config
    from opentpu.kernels.cactus_cq import cq_activation
    from opentpu.runtime import Input, Output, compile_kernel, run

    path = tmp_path / "synthetic.cact"
    fixture(path, bits, inter)
    source = read_cact(path)
    folded, packed, scales, lut = cq_to_tmm_cq(source)
    n, k = folded.indices.shape
    assert packed.dtype == np.uint8 and packed.shape[0] == n and packed.shape[1] % 128 == 0
    assert scales.dtype == np.float32 and scales.shape == (n, k // 128)
    assert lut.dtype == np.int8 and lut.size == 1 << bits
    assert np.array_equal(folded.permutation, np.arange(128))

    weight = QTensor.cq_weight(0x20000, 0x30000, n, k, 128, bits, lut, rs=packed.shape[1])
    x = np.random.default_rng(1181 + bits + inter).normal(size=(2, k)).astype(np.float32)
    left = np.tile(folded.left_signs, k // 128).astype(np.float32)
    right = np.tile(folded.right_signs, k // 128).astype(np.float32)

    @ol.jit
    def projection(src, reciprocal, lsign, rsign, dst, weight):
        basis = cq_activation(ol.load(src), ol.load(reciprocal), ol.load(lsign),
                              ol.load(rsign), tuple(range(128)))
        ol.store(dst, ol.dot(basis, weight))

    cfg = Config(D=128, MCOLS=4, DRAM_BYTES=1 << 20, TGSC=True, TCQ=True, VBFLY=True)
    comp, images = compile_kernel(projection, cfg, src=Input(x),
                                  reciprocal=Input(folded.input_scale_recip), lsign=Input(left),
                                  rsign=Input(right), dst=Output((2, n)), weight=weight)
    instructions = [ins for program in comp.programs for ins in program]
    assert any(ins.op == I.TMM and ins.w[3] & I.TMM_CQ for ins in instructions)
    assert any(ins.op == I.VOP and ((ins.w[5] >> 16) & 255) == I.V_BFLY for ins in instructions)
    for addr, value in ((0x20000, packed), (0x30000, scales)):
        data = np.ascontiguousarray(value).view(np.uint8).reshape(-1)
        images[0][addr:addr + data.size] = data
    got = run(comp, images, "isa").outputs["dst"]
    expected = source.matmul_reference(x)
    assert np.linalg.norm(got - expected) / np.linalg.norm(expected) < 0.035
    device_expected = folded.transform(x).reshape(x.shape) @ device_cq.device_dense(
        folded.indices, folded.norms, folded.codebook).T
    assert np.linalg.norm(got - device_expected) / np.linalg.norm(device_expected) < 0.015
    if bits == 2 and not inter:
        for config, message in ((replace(cfg, TCQ=False), "TCQ"), (replace(cfg, TGSC=False), "TGSC")):
            with pytest.raises(CompileError, match=message):
                compile_kernel(projection, config, src=Input(x), reciprocal=Input(folded.input_scale_recip),
                               lsign=Input(left), rsign=Input(right), dst=Output((2, n)), weight=weight)


def test_cact_native_conversion_rejects_invalid_contracts(tmp_path):
    from opentpu.cactus.transcode import cq_to_tmm_cq
    path = tmp_path / "synthetic.cact"
    fixture(path, 2, False)
    source = read_cact(path)
    for broken in (replace(source, group_size=64), replace(source, bits=3),
                   replace(source, norms=source.norms[:, :-1]),
                   replace(source, codebook=np.array([0.0, float("nan"), 0.0, 1.0], np.float32)),
                   replace(source, codebook=np.full(4, np.nextafter(np.float32(0), np.float32(1)), np.float32)),
                   replace(source, permutation=np.zeros(128, np.uint32)),
                   replace(source, indices=np.full_like(source.indices, 4)),
                   replace(source, input_scale_recip=np.zeros_like(source.input_scale_recip)),
                   replace(source, input_scale=np.zeros_like(source.input_scale))):
        with pytest.raises(ValueError):
            cq_to_tmm_cq(broken)

    zero = replace(source, codebook=np.zeros(4, np.float32))
    _, packed, scales, lut = cq_to_tmm_cq(zero)
    assert packed.size > 0 and np.isfinite(scales).all() and np.count_nonzero(lut) == 0
