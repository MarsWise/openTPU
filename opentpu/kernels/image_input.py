"""Device-side layout operations for image inputs already held in DRAM."""
from .. import isa as I


def patchify_programs(cfg, src: int, dst: int, height: int, width: int,
                      patch: int = 14, channels: int = 3, order: str = 'raster',
                      dst_stride: int | None = None):
    """Convert contiguous fp32 HWC into [patches, C * patch * patch] rows.

    ``raster`` outputs row-major grid order. ``minicpm`` outputs the tower's
    4x4 block / 2x2 window order; its patch grid dimensions must be multiples
    of four. Inside each patch the order is channel, row, column. Return
    complete IMEM-sized programs; the caller runs them in order. The source
    and destination DRAM ranges are disjoint. ``dst_stride`` is the distance
    between destination rows in fp32 words; padding is preserved, so callers
    must initialize it to the value expected by their model.
    """
    if not all(isinstance(x, int) for x in (src, dst, height, width, patch, channels)):
        raise TypeError("addresses and dimensions must be integers")
    if (src < 0 or dst < 0 or src % 4 or dst % 4 or height <= 0 or width <= 0
            or patch <= 0 or channels not in (1, 3, 4)
            or height % patch or width % patch):
        raise ValueError("invalid addresses or image, patch and channel dimensions")
    grid_h, grid_w = height // patch, width // patch
    if order not in ('raster', 'minicpm'):
        raise ValueError("patch order must be raster or minicpm")
    if order == 'minicpm' and (grid_h % 4 or grid_w % 4):
        raise ValueError("minicpm patch grid must be divisible by four")
    patch_words = channels * patch * patch
    dst_stride = patch_words if dst_stride is None else dst_stride
    if not isinstance(dst_stride, int) or dst_stride < patch_words:
        raise ValueError("destination stride must cover a whole patch row")
    src_size = 4 * height * width * channels
    dst_size = 4 * ((grid_h * grid_w - 1) * dst_stride + patch_words)
    if src + src_size > cfg.DRAM_BYTES or dst + dst_size > cfg.DRAM_BYTES:
        raise ValueError("image or patches outside DRAM")
    if src < dst + dst_size and dst < src + src_size:
        raise ValueError("image and patches must not overlap")
    row_words = channels * patch
    if patch_words + row_words > cfg.TMEM_WORDS or patch >= 65536:
        raise ValueError("patch and row scratch do not fit TMEM")
    # One LD and one gather per channel per row, then a single ST per patch.
    per_patch = patch * (1 + channels) + 1
    per_program = (cfg.IMEM_WORDS // 8 - 1) // per_patch
    if per_program < 1:
        raise ValueError("one patch program exceeds IMEM")
    if order == 'raster':
        coords = ((y, x) for y in range(grid_h) for x in range(grid_w))
    else:
        coords = ((4 * by + 2 * wy + ly, 4 * bx + 2 * wx + lx)
                  for by in range(grid_h // 4) for bx in range(grid_w // 4)
                  for wy in range(2) for wx in range(2)
                  for ly in range(2) for lx in range(2))
    programs, out, index = [], [], 0
    for gy, gx in coords:
        y, x = gy * patch, gx * patch
        for py in range(patch):
            row_addr = src + 4 * ((y + py) * width + x) * channels
            out.append(I.ld(row_addr, patch_words, row_words))
            for c in range(channels):
                out.append(I.vop(I.V_COPY, c * patch * patch + py * patch,
                                 patch_words + c, 0, patch, 1, 1, channels, 0))
        out.append(I.st(dst + 4 * index * dst_stride, 0, patch_words))
        index += 1
        if index % per_program == 0:
            out.append(I.halt())
            programs.append(out)
            out = []
    if out:
        out.append(I.halt())
        programs.append(out)
    return programs
