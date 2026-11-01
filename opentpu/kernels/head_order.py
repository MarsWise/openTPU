"""Device-side head ordering for grouped output projections."""
from .. import language as ol


def tiled_to_grouped(x, nk: int, rep: int, head_dim: int):
    """Copy [rows, rep, nk, head_dim] into [rows, nk, rep, head_dim].

    Input and output are tiles; all copies execute on the device. The caller must
    apply this before any grouped-order signs or Hadamard transform.
    """
    if nk <= 0 or rep <= 0 or head_dim <= 0 or x.cols != nk * rep * head_dim:
        raise ValueError("invalid DeltaNet head geometry")
    out = ol.empty(x.shape, dense=True)
    for key in range(nk):
        for repeat in range(rep):
            src = (repeat * nk + key) * head_dim
            dst = (key * rep + repeat) * head_dim
            out[:, dst:dst + head_dim].set(x[:, src:src + head_dim])
    return out
