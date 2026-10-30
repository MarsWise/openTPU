"""Convert raw packed unsigned input bytes to fp32 entirely on the device.

The host writes only raw bytes and allocates up to three padding bytes at the end.
One VOP extracts each byte position from raw TMEM words, then four strided VOP
copies restore the original byte order. Results are exact fp32 integers 0..255.
Image resize, channel layout and normalization are separate operations.
"""
from .. import isa as I
import math


def program(cfg, src: int, dst: int, count: int, tmem_base: int = 0):
    """Return a complete program reading DRAM bytes and writing count fp32 words.

    src and dst must be word aligned and their DRAM ranges must not overlap.
    The source allocation includes zero to three extra padding bytes. TMEM starting
    at tmem_base is scratch. Large inputs are processed as independent chunks.
    """
    if not cfg.VU8:
        raise ValueError("packed u8 input needs Config.VU8 / new firmware")
    if not all(isinstance(x, int) for x in (src, dst, count, tmem_base)):
        raise TypeError("addresses and count must be integers")
    if count <= 0 or src < 0 or dst < 0 or tmem_base < 0 or src % 4 or dst % 4:
        raise ValueError("count must be positive; DRAM addresses must be word aligned")
    padded = (count + 3) & ~3
    if src + padded > cfg.DRAM_BYTES or dst + 4 * count > cfg.DRAM_BYTES:
        raise ValueError("input or output outside DRAM")
    if src < dst + 4 * count and dst < src + padded:
        raise ValueError("input and output DRAM ranges must not overlap")
    nmax = min(65535, (cfg.TMEM_WORDS - tmem_base) // 9)
    if nmax < 1:
        raise ValueError("not enough TMEM scratch")
    out = []
    for first in range(0, padded // 4, nmax):
        n = min(nmax, padded // 4 - first)
        actual = min(4 * n, count - 4 * first)
        raw = tmem_base
        tmp = raw + n
        result = tmp + 4 * n
        out.append(I.ld(src + 4 * first, raw, n))
        for byte in range(4):
            out.append(I.vop(I.V_U8[byte], tmp + byte * n, raw, 0,
                             1, n, n, n, 0))
        for byte in range(4):
            out.append(I.vop(I.V_COPY, result + byte, tmp + byte * n, 0,
                             n, 1, 4, 1, 0))
        out.append(I.st(dst + 16 * first, result, actual))
    out.append(I.halt())
    if len(out) * 8 > cfg.IMEM_WORDS:
        raise ValueError("u8 input program exceeds IMEM; split the input into calls")
    return out


def program_rows(cfg, src: int, dst: int, scratch: int, rows: int,
                 row_words: int, dst_stride: int):
    """Convert dense uint8 rows into strided fp32 rows in DRAM.

    scratch holds the contiguous fp32 conversion and is disjoint from both source
    and destination. Destination padding is preserved; all strides are in words.
    """
    if rows <= 0 or row_words <= 0 or dst_stride < row_words:
        raise ValueError("invalid row count, width or destination stride")
    count = rows * row_words
    if dst % 4 or scratch % 4 or dst < 0 or scratch < 0:
        raise ValueError("destination and scratch must be word-aligned DRAM addresses")
    dst_end = dst + 4 * ((rows - 1) * dst_stride + row_words)
    scratch_end = scratch + 4 * count
    src_end = src + ((count + 3) & ~3)
    if dst_end > cfg.DRAM_BYTES or scratch_end > cfg.DRAM_BYTES:
        raise ValueError("strided destination or scratch outside DRAM")
    if ((dst < scratch_end and scratch < dst_end) or
            (dst < src_end and src < dst_end)):
        raise ValueError("destination overlaps input or scratch")
    out = program(cfg, src, scratch, count)[:-1]
    # Copy each dense row into its destination; the gap is never read or written.
    for r in range(rows):
        for c in range(0, row_words, cfg.TMEM_WORDS):
            n = min(cfg.TMEM_WORDS, row_words - c)
            out.append(I.ld(scratch + 4 * (r * row_words + c), 0, n))
            out.append(I.st(dst + 4 * (r * dst_stride + c), 0, n))
    out.append(I.halt())
    if len(out) * 8 > cfg.IMEM_WORDS:
        raise ValueError("strided u8 program exceeds IMEM; split the input into calls")
    return out


def program_normalize(cfg, src: int, dst: int, pixels: int, scale, bias):
    """Convert packed HWC uint8 and apply per-channel scale and bias on device.

    ``scale`` and ``bias`` contain one fp32 constant per channel, in the same
    order as the input bytes. Output is contiguous HWC fp32. Both ranges must
    be disjoint and the packed source needs up to three bytes of padding.
    Computation is two fp32 operations, multiply followed by add per channel.
    """
    if not isinstance(pixels, int) or pixels <= 0:
        raise ValueError("pixels must be a positive integer")
    scale, bias = tuple(scale), tuple(bias)
    channels = len(scale)
    if channels not in (1, 3, 4) or len(bias) != channels:
        raise ValueError("scale and bias must have 1, 3 or 4 channels")
    if not all(math.isfinite(float(x)) for x in (*scale, *bias)):
        raise ValueError("scale and bias must be finite")
    out = program(cfg, src, dst, pixels * channels)[:-1]
    chunk = min(cfg.TMEM_WORDS // channels, 65535)
    if chunk < 1:
        raise ValueError("not enough TMEM scratch for a pixel")
    for first in range(0, pixels, chunk):
        n = min(chunk, pixels - first)
        addr = dst + 4 * first * channels
        out.append(I.ld(addr, 0, n * channels))
        for c, (mul, add) in enumerate(zip(scale, bias)):
            out.append(I.vop(I.V_MUL, c, c, 0, n, 1, channels, channels, 0,
                             I.B_SCALAR, float(mul)))
            out.append(I.vop(I.V_ADD, c, c, 0, n, 1, channels, channels, 0,
                             I.B_SCALAR, float(add)))
        out.append(I.st(addr, 0, n * channels))
    out.append(I.halt())
    if len(out) * 8 > cfg.IMEM_WORDS:
        raise ValueError("normalized u8 program exceeds IMEM; split the input into calls")
    return out
