"""Experimental four-value INT6 byte layout, not a QST or model format."""
import numpy as np


def quantize_rows(values):
    """Experimental symmetric fp32 row quantization, matching the QST arithmetic style.

    Finite inputs only; the proposed arithmetic and resulting bytes have not
    yet been integrated into RTL QST or qualified for KV model quality.
    """
    from opentpu import fp32 as F
    if (not isinstance(values, np.ndarray) or values.dtype != np.float32 or
            values.ndim != 2 or min(values.shape) <= 0 or not np.isfinite(values).all()):
        raise ValueError("quantize_rows needs a nonempty finite fp32 matrix")
    x = F.ftz(values)
    amax = F.from_bits(np.max(F.bits(F.fabs(x)), axis=1))
    zero = amax == 0
    if np.any((~zero) & ((amax < np.float32(2.0 ** -121)) | (amax >= np.float32(2.0 ** 126)))):
        raise ValueError("int6 reference row magnitude outside validated fp32 range")
    scale = np.where(zero, np.float32(0), F.mul(amax, np.float32(1.0 / 31.0))).astype(np.float32)
    inv = np.where(zero, np.float32(0), F.mul(np.float32(31), F.recip(amax))).astype(np.float32)
    y = F.mul(x, inv[:, None])
    q = np.clip(np.rint(y), -31, 31).astype(np.int8)
    return q, scale


def pack4(values):
    """Pack four signed int8 values in [-32,31] into three little-endian bytes."""
    if not isinstance(values, np.ndarray) or values.dtype != np.int8 or not values.ndim or values.shape[-1] != 4:
        raise ValueError("pack4 needs int8 arrays with four values per group")
    if np.any(values < -32) or np.any(values > 31):
        raise ValueError("int6 value outside [-32,31]")
    u = values.astype(np.int32) & 63
    word = u[..., 0] | (u[..., 1] << 6) | (u[..., 2] << 12) | (u[..., 3] << 18)
    return np.stack((word & 255, (word >> 8) & 255, (word >> 16) & 255), axis=-1).astype(np.uint8)


def unpack4(data):
    """Decode three little-endian bytes into four sign-extended int8 values."""
    if not isinstance(data, np.ndarray) or data.dtype != np.uint8 or not data.ndim or data.shape[-1] != 3:
        raise ValueError("unpack4 needs uint8 arrays with three bytes per group")
    b = data.astype(np.int32)
    word = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)
    code = np.stack(tuple((word >> n) & 63 for n in (0, 6, 12, 18)), axis=-1)
    return np.where(code >= 32, code - 64, code).astype(np.int8)


def pack_rows(values):
    """Pack complete 16-element groups; standalone proposed QST6 row layout."""
    if (not isinstance(values, np.ndarray) or values.dtype != np.int8 or values.ndim != 2 or
            not values.shape[0] or not values.shape[1] or values.shape[1] % 16):
        raise ValueError("packed rows require a nonempty int8 matrix with width a multiple of 16")
    rows, width = values.shape
    return pack4(values.reshape(rows, width // 4, 4)).reshape(rows, width * 3 // 4)


def unpack_rows(data):
    """Unpack rows containing complete 12-byte groups (16 values each)."""
    if (not isinstance(data, np.ndarray) or data.dtype != np.uint8 or data.ndim != 2 or
            not data.shape[0] or not data.shape[1] or data.shape[1] % 12):
        raise ValueError("packed rows require a nonempty uint8 matrix with width a multiple of 12")
    rows, width = data.shape
    return unpack4(data.reshape(rows, width // 3, 3)).reshape(rows, width * 4 // 3)
