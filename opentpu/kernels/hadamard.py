"""Prism's Hadamard activation transform executed entirely by the FPGA VPU."""
import math

from .. import language as ol


def hadamard(x, signs, block=512, inverse=False):
    if block <= 0 or block & (block - 1) or x.cols % block:
        raise ValueError("Hadamard block must be a power of two dividing the width")
    if signs.shape != (x.cols,):
        raise ValueError("Hadamard signs must match the activation width")
    y = ol.empty(x.shape, dense=True)
    y.set(x if inverse else x * signs[None, :])
    shape = x.shape
    for log_stride in range(block.bit_length() - 1):
        stride = 1 << log_stride
        a = y.reshape(y.rows * y.cols // (2 * stride), 2 * stride)
        out = ol.empty(a.shape, dense=True)
        out[:, :stride].set(a[:, :stride] + a[:, stride:])
        out[:, stride:].set(a[:, :stride] - a[:, stride:])
        y = out.reshape(*shape)
    y = y * (1 / math.sqrt(block))
    return y * signs[None, :] if inverse else y


def stationary(x, weight):
    """Rotate once for several projections sharing the same activation and sign vector."""
    from ..compiler import Tensor
    source = x.src if hasattr(x, "src") else x
    signs = ol.load(Tensor(weight.hsigns, (weight.shape[1],), (1,)))
    return ol.quantize(hadamard(source, signs, weight.hblock))
