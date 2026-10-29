"""Convert raw packed unsigned input bytes to fp32 entirely on the device.

The host writes only raw bytes and allocates up to three padding bytes at the end.
One VOP extracts each byte position from raw TMEM words, then four strided VOP
copies restore the original byte order. Results are exact fp32 integers 0..255.
Image resize, channel layout and normalization are separate operations.
"""
from .. import isa as I


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
