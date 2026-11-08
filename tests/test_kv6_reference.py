import numpy as np
import pytest

from opentpu.nextgen.kv6 import pack4, unpack4, pack_rows, unpack_rows, quantize_rows


def test_fixed_signed_example_and_roundtrip():
    values = np.array([[0, -1, -32, 31], [31, -32, 0, -1]], np.int8)
    packed = pack4(values)
    assert packed[0].tolist() == [0xc0, 0x0f, 0x7e]
    np.testing.assert_array_equal(unpack4(packed), values)


def test_all_codes_independently_in_each_lane():
    codes = np.zeros((256, 4), np.int8)
    for lane in range(4):
        codes[:, lane] = np.tile(np.arange(-32, 32, dtype=np.int8), 4)
        np.testing.assert_array_equal(unpack4(pack4(codes)), codes)
        codes[:, lane] = 0


def test_invalid_values_and_shapes_fail_closed():
    for x in (np.array([[32, 0, 0, 0]], np.int8), np.array([[-33, 0, 0, 0]], np.int8),
              np.zeros((2, 5), np.int8), np.zeros((2, 4), np.float32)):
        with pytest.raises(ValueError):
            pack4(x)
    with pytest.raises(ValueError):
        unpack4(np.zeros((1, 4), np.uint8))


def test_symmetric_quantization_produces_roundtrip_and_zero():
    x = np.array([[0., 1., -1., .5], [0., 0., 0., 0.]], np.float32)
    q, scale = quantize_rows(x)
    assert q.tolist() == [[0, 31, -31, 16], [0, 0, 0, 0]]
    np.testing.assert_allclose(scale, [1 / 31, 0], rtol=1e-6)
    np.testing.assert_array_equal(unpack4(pack4(q)), q)


def test_round_to_even_ties_and_adjacent_fp32_values():
    # The anchor makes the ideal inverse scale 1, so half-integers and their
    # neighbouring representable floats have independently known results.
    ties = np.arange(0.5, 31.0, 1.0, dtype=np.float32)
    below = np.nextafter(ties, np.float32(-np.inf))
    above = np.nextafter(ties, np.float32(np.inf))
    values = np.concatenate(([31.0], ties, below, above, -ties, -below, -above)).astype(np.float32)[None, :]
    q, scale = quantize_rows(values)
    rounded_even = np.rint(ties).astype(np.int8)
    lower = np.floor(ties).astype(np.int8)
    upper = np.ceil(ties).astype(np.int8)
    expected = np.concatenate(([31], rounded_even, lower, upper,
                               -rounded_even, -lower, -upper)).astype(np.int8)
    np.testing.assert_array_equal(q[0], expected)
    np.testing.assert_array_equal(scale, np.array([1.0], np.float32))


@pytest.mark.parametrize("x", [np.array([[np.nan]], np.float32),
                               np.array([[np.inf]], np.float32),
                               np.array([[1]], np.float64),
                               np.zeros((1, 0), np.float32)])
def test_quantization_rejects_unsupported_input(x):
    with pytest.raises(ValueError):
        quantize_rows(x)


def test_quantization_on_seeded_rows_against_float_input():
    x = np.random.default_rng(923).normal(size=(32, 128)).astype(np.float32)
    q, scale = quantize_rows(x)
    restored = q.astype(np.float32) * scale[:, None]
    assert np.max(np.abs(q.astype(np.int16))) <= 31
    assert np.linalg.norm(restored - x) / np.linalg.norm(x) < .04
    np.testing.assert_array_equal(unpack4(pack4(q.reshape(32, 32, 4))).reshape(x.shape), q)


@pytest.mark.parametrize("amplitude", [np.float32(2.0 ** 126), np.float32(2.0 ** -125)])
def test_quantization_refuses_unsafe_fp32_range(amplitude):
    with pytest.raises(ValueError, match="range"):
        quantize_rows(np.array([[amplitude, -amplitude]], np.float32))


def test_quantization_accepts_safe_fp32_boundaries_and_flushes_subnormals():
    tiny, large = np.float32(2.0 ** -121), np.nextafter(np.float32(2.0 ** 126), np.float32(0))
    for amplitude in (tiny, large):
        q, scale = quantize_rows(np.array([[amplitude, -amplitude]], np.float32))
        assert scale[0] > 0 and np.isfinite(scale[0])
        assert q.tolist() == [[31, -31]]
    q, scale = quantize_rows(np.array([[np.nextafter(np.float32(0), np.float32(1)), 0]], np.float32))
    assert q.tolist() == [[0, 0]] and scale.tolist() == [0]


def test_adjacent_safe_and_rejected_magnitudes():
    minimum = np.float32(2.0 ** -121)
    maximum_exclusive = np.float32(2.0 ** 126)
    for amplitude in (np.nextafter(minimum, np.float32(np.inf)),
                      np.nextafter(maximum_exclusive, np.float32(0))):
        q, scale = quantize_rows(np.array([[amplitude, -amplitude]], np.float32))
        assert q.tolist() == [[31, -31]]
        assert np.isfinite(scale[0]) and scale[0] > 0
    for amplitude in (np.nextafter(minimum, np.float32(0)), maximum_exclusive):
        with pytest.raises(ValueError, match="range"):
            quantize_rows(np.array([[amplitude, -amplitude]], np.float32))


@pytest.mark.parametrize("width", [16, 32, 128, 256])
def test_complete_rows_follow_little_endian_contiguous_contract(width):
    values = np.random.default_rng(width).integers(-32, 32, (3, width), dtype=np.int8)
    packed = pack_rows(values)
    assert packed.shape == (3, width * 3 // 4)
    expected = bytearray()
    for offset in range(0, width, 4):
        group = values[0, offset:offset + 4]
        word = sum((int(v) & 63) << (6 * lane) for lane, v in enumerate(group))
        expected.extend(word.to_bytes(3, 'little'))
    assert packed[0].tobytes() == expected
    np.testing.assert_array_equal(unpack_rows(packed), values)


def test_rows_reject_partial_or_invalid_groups():
    for bad in (np.zeros((1, 15), np.int8), np.zeros((1, 20), np.int8),
                np.zeros((0, 16), np.int8), np.zeros((1, 16), np.float32)):
        with pytest.raises(ValueError):
            pack_rows(bad)
    for bad in (np.zeros((1, 13), np.uint8), np.zeros((0, 12), np.uint8)):
        with pytest.raises(ValueError):
            unpack_rows(bad)
