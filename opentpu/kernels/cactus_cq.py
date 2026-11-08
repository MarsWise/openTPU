"""Cactus CQ activation basis on the existing VPU.

Experimental primitive: this is not a CQ matrix multiplication or a model
backend. Permutation is specialized at compile time; arithmetic is on device.
"""
from .. import language as ol
from .hadamard import hadamard


def cq_activation(x, reciprocal, left, right, permutation, group_size=128):
    """Compute (x * reciprocal * left) @ normalized_H, then right and permute.

    The 1-D reciprocal and sign tiles span the full activation width; callers
    repeat the group signs offline. `permutation` is the fixed group-local
    checkpoint permutation. Both 1-group and multi-group inputs are supported.
    """
    if len(x.shape) != 2 or group_size <= 0 or group_size & (group_size - 1):
        raise ValueError('CQ activation requires a 2-D tile and power-of-two group')
    if x.cols % group_size or any(t.shape != (x.cols,) for t in (reciprocal, left, right)):
        raise ValueError('CQ scales and signs must span the activation width')
    if len(permutation) != group_size or sorted(permutation) != list(range(group_size)):
        raise ValueError('invalid CQ group permutation')

    basis = hadamard(x * reciprocal[None, :], left, block=group_size)
    basis = basis * right[None, :]
    if all(i == p for i, p in enumerate(permutation)):
        return basis
    result = ol.empty(x.shape, dense=True)
    for base in range(0, x.cols, group_size):
        for dest, src in enumerate(permutation):
            result[:, base + dest:base + dest + 1].set(basis[:, base + src:base + src + 1])
    return result
