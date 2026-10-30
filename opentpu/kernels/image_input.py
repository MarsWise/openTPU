"""Device-side layout operations for image inputs already held in DRAM."""
from .. import isa as I


def patchify_programs(cfg, src: int, dst: int, height: int, width: int,
                      patch: int = 14, channels: int = 3):
    """Convert contiguous fp32 HWC into [patches, C * patch * patch] rows.

    Patches are in row-major grid order; inside each patch the order is
    channel, row, column. Return complete IMEM-sized programs; the caller
    runs them in order. The source and destination DRAM ranges are disjoint.
    """
    if not all(isinstance(x, int) for x in (src, dst, height, width, patch, channels)):
        raise TypeError("addresses and dimensions must be integers")
    if (src < 0 or dst < 0 or src % 4 or dst % 4 or height <= 0 or width <= 0
            or patch <= 0 or channels not in (1, 3, 4)
            or height % patch or width % patch):
        raise ValueError("invalid addresses or image, patch and channel dimensions")
    nwords = height * width * channels
    size = 4 * nwords
    if src + size > cfg.DRAM_BYTES or dst + size > cfg.DRAM_BYTES:
        raise ValueError("image or patches outside DRAM")
    if src < dst + size and dst < src + size:
        raise ValueError("image and patches must not overlap")
    patch_words = channels * patch * patch
    row_words = channels * patch
    if patch_words + row_words > cfg.TMEM_WORDS or patch >= 65536:
        raise ValueError("patch and row scratch do not fit TMEM")
    # One LD and one gather per channel per row, then a single ST per patch.
    per_patch = patch * (1 + channels) + 1
    per_program = (cfg.IMEM_WORDS // 8 - 1) // per_patch
    if per_program < 1:
        raise ValueError("one patch program exceeds IMEM")
    programs, out, index = [], [], 0
    for y in range(0, height, patch):
        for x in range(0, width, patch):
            for py in range(patch):
                row_addr = src + 4 * ((y + py) * width + x) * channels
                out.append(I.ld(row_addr, patch_words, row_words))
                for c in range(channels):
                    out.append(I.vop(I.V_COPY, c * patch * patch + py * patch,
                                     patch_words + c, 0, patch, 1, 1, channels, 0))
            out.append(I.st(dst + 4 * index * patch_words, 0, patch_words))
            index += 1
            if index % per_program == 0:
                out.append(I.halt())
                programs.append(out)
                out = []
    if out:
        out.append(I.halt())
        programs.append(out)
    return programs
