import numpy as np
import pytest

from opentpu.quant import pack_ternary, ternary_matmul, unpack_ternary


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
