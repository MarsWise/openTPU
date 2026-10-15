import numpy as np
import pytest

from opentpu import isa as I
from opentpu.compiler import Builder, QTensor
from opentpu.fp32 import i2f, mul as fmul
from opentpu.isasim import Config, Machine, SimError
from opentpu.quant import pack_ternary, pack_ternary_mxu, ternary_matmul, unpack_ternary


@pytest.mark.parametrize("count", range(10))
def test_ternary_packing_round_trip_and_tail(count):
    values = np.resize(np.array([-1, 0, 1], dtype=np.int8), count)
    packed = pack_ternary(values)
    assert packed.shape == ((count + 3) // 4,)
    assert unpack_ternary(packed, count).tolist() == values.tolist()
    if count % 4:
        assert packed[-1] >> (2 * (count % 4)) == 0


def test_ternary_packing_is_low_pair_first_and_supports_rows():
    values = np.array([[0, 1, -1, 0, 1], [-1, -1, 0, 1, 0]], dtype=np.int8)
    packed = pack_ternary(values)
    assert packed.tolist() == [[0b00100100, 0b00000001], [0b01001010, 0]]
    assert np.array_equal(unpack_ternary(packed, values.shape[-1]), values)


@pytest.mark.parametrize("packed,count,message", [
    (np.array([0b11], dtype=np.uint8), 1, "reserved"),
    (np.array([0b0100], dtype=np.uint8), 1, "padding"),
])
def test_ternary_unpack_rejects_invalid_symbols_and_padding(packed, count, message):
    with pytest.raises(ValueError, match=message):
        unpack_ternary(packed, count)


def test_ternary_unpack_rejects_inconsistent_shapes_and_types():
    with pytest.raises(ValueError, match="length"):
        unpack_ternary(np.array([0], dtype=np.uint8), 5)
    with pytest.raises(TypeError, match="uint8"):
        unpack_ternary(np.array([0], dtype=np.int8), 1)
    with pytest.raises(ValueError, match="count"):
        unpack_ternary(np.array([0], dtype=np.uint8), -1)
    with pytest.raises(ValueError, match="dimension"):
        pack_ternary(np.array(1, dtype=np.int8))
    with pytest.raises(TypeError, match="integer"):
        pack_ternary(np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="must be in"):
        pack_ternary(np.array([-2, 0, 1], dtype=np.int8))


@pytest.mark.parametrize("k", [1, 3, 4, 5, 7, 8, 17])
def test_ternary_matmul_matches_scalar_reference(k):
    rng = np.random.default_rng(0x7E2 + k)
    activations = rng.integers(-128, 128, size=(3, k), dtype=np.int8)
    weights = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(5, k))
    got = ternary_matmul(activations, pack_ternary(weights))
    expected = np.empty((3, 5), dtype=np.int64)
    for row in range(3):
        for col in range(5):
            expected[row, col] = sum(int(activations[row, i]) * int(weights[col, i])
                                     for i in range(k))
    assert got.dtype == np.int64
    assert np.array_equal(got, expected)


def test_ternary_matmul_keeps_leading_batch_dimensions_and_checks_shapes():
    activations = np.array([[[127, -128, 1], [-1, 0, 4]]], dtype=np.int16)
    weights = np.array([[1, -1, 0], [-1, 1, 1]], dtype=np.int8)
    result = ternary_matmul(activations, pack_ternary(weights))
    assert result.shape == (1, 2, 2)
    assert result.tolist() == [[[255, -254], [-1, 5]]]
    with pytest.raises(TypeError, match="signed integer"):
        ternary_matmul(np.array([1.0]), pack_ternary(weights))
    with pytest.raises(ValueError, match="packed_weights"):
        ternary_matmul(np.array([1], dtype=np.int8), pack_ternary(weights[0]))


def test_ternary_matmul_rejects_reserved_codes_and_nonzero_padding():
    activations = np.array([1], dtype=np.int8)
    with pytest.raises(ValueError, match="reserved"):
        ternary_matmul(activations, np.array([[0b11]], dtype=np.uint8))
    with pytest.raises(ValueError, match="padding"):
        ternary_matmul(activations, np.array([[0b0100]], dtype=np.uint8))


def test_ternary_mxu_packing_pads_each_row_to_a_stream_chunk():
    weights = np.array([[1, 0] * 20, [-1, 0, 1, 0] * 10], dtype=np.int8)
    packed = pack_ternary_mxu(weights, D=8)
    assert packed.shape == (2, 16)
    assert np.array_equal(unpack_ternary(packed[:, :10], weights.shape[1]), weights)
    assert np.all(packed[:, 10:] == 0)


@pytest.mark.parametrize("unit", [False, True])
def test_tmm_simulator_matches_grouped_integer_reference(unit):
    rng = np.random.default_rng(0x7E2)
    cfg = Config(S=1, D=8, MCOLS=4, ACT_BLOCKS=8, TMEM_WORDS=256, DRAM_BYTES=4096,
                 LANES=4)
    M, N, KB, ab, D, ors = 3, 5, 7, 1, cfg.D, 7
    activations = rng.integers(-128, 128, size=(M, KB, D), dtype=np.int8)
    offsets = rng.integers(0, 8, size=(M, KB, 4), dtype=np.uint8)
    sa = np.array([0.25, 0.5, 1.0], dtype=np.float32)
    weights = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(N, KB * D))
    packed = pack_ternary_mxu(weights, D)
    rs, weight_addr, scale_addr, out = packed.shape[1], 128, 512, 32
    dram = np.zeros(cfg.DRAM_BYTES, dtype=np.uint8)
    dram[weight_addr:weight_addr + packed.size] = packed.reshape(-1)
    sw = np.ones(N, dtype=np.float32) if unit else np.linspace(0.25, 1.25, N, dtype=np.float32)
    if not unit:
        dram.view(np.float32)[scale_addr // 4:scale_addr // 4 + N] = sw

    ins = I.tmm(weight_addr, scale_addr, out, N, KB, rs, ors, M, ab, 4,
                unit=unit, rsc=not unit)
    assert I.Instr.decode(ins.encode()).op == I.TMM
    machine = Machine(cfg, [[ins, I.halt()]], [dram])
    sl = machine.slices[0]
    sl.act[:M, ab * D:(ab + KB) * D] = activations.reshape(M, KB * D)
    sl.ascale[:M, ab:ab + KB] = sa[:, None]
    sl.adoff[:M, ab:ab + KB] = offsets
    machine.run()

    acc = np.zeros((M, N), dtype=np.int64)
    for k in range(KB):
        for g in range(4):
            lo, hi = g * (D // 4), (g + 1) * (D // 4)
            dot = (activations[:, k, lo:hi].astype(np.int64) @
                   weights[:, k * D + lo:k * D + hi].astype(np.int64).T)
            shift = 11 - offsets[:, k, g].astype(np.int64)
            acc += np.where(shift[:, None] >= 0,
                            dot << np.maximum(shift[:, None], 0), 0)
    expected = fmul(fmul(i2f(acc), sw[None, :]), fmul(sa[:, None], np.float32(2.0 ** -11)))
    got = sl.tget(out + np.arange(M)[:, None] * ors + np.arange(N)[None, :])
    assert np.array_equal(got.view(np.uint32), expected.view(np.uint32))


def test_tmm_rejects_reserved_weight_symbols():
    cfg = Config(S=1, D=8, MCOLS=2, ACT_BLOCKS=2, TMEM_WORDS=64, DRAM_BYTES=1024, LANES=2)
    weights = pack_ternary_mxu(np.zeros((1, 8), dtype=np.int8), cfg.D)
    weights[0, 0] = 0b11
    dram = np.zeros(cfg.DRAM_BYTES, dtype=np.uint8)
    dram[64:64 + weights.shape[1]] = weights[0]
    machine = Machine(cfg, [[I.tmm(64, 0, 0, 1, 1, 8, 1, 1, 0, 0, unit=True),
                             I.halt()]], [dram])
    with pytest.raises(SimError, match="reserved ternary code"):
        machine.run()


def test_compiler_dot_lowers_ternary_weight_descriptor_to_tmm():
    cfg = Config(S=1, D=8, MCOLS=4, ACT_BLOCKS=8, TMEM_WORDS=256, DRAM_BYTES=4096,
                 LANES=4)
    builder = Builder(cfg, 0)
    activations = builder.alloc((3, 7 * cfg.D))
    weights = QTensor.ternary_weight(0x1000, None, 5, 7 * cfg.D, cfg.D)
    builder.dot(activations, weights)
    assert [ins.op for ins in builder.root].count(I.TMM) == 1
    tmm = next(ins for ins in builder.root if ins.op == I.TMM)
    assert (tmm.w[3] & 0xFFFF, (tmm.w[3] >> 16) & 0xFF) == (5, 7)
    assert tmm.flags & I.F_UNIT
