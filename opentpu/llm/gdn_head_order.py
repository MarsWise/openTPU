"""Offline conversion from tiled GGUF GDN value-head rows to grouped order.

Q/K rows and the grouped Hadamard-folded output projection are not reordered.
This helper is not yet wired into a model loader or FPGA execution path.
"""
from collections.abc import Mapping
from dataclasses import replace

import numpy as np

_VALUE_INPUTS = ("linear_attn.in_proj_qkv.weight", "linear_attn.conv1d.weight",
                 "linear_attn.in_proj_z.weight", "linear_attn.in_proj_a.weight",
                 "linear_attn.in_proj_b.weight", "linear_attn.A_log", "linear_attn.dt_bias")


def grouped_indices(nk: int, nv: int):
    if nk <= 0 or nv <= 0 or nv % nk:
        raise ValueError("invalid DeltaNet head counts")
    return np.arange(nv).reshape(nv // nk, nk).T.reshape(-1)


def reorder_gdn_tensor(name: str, a, nk: int, nv: int, dk: int, dv: int):
    """Reorder one named input tensor's leading value-head axis without requantizing."""
    idx = grouped_indices(nk, nv)
    x = np.asarray(a)
    if x.ndim < 1 or dk <= 0 or dv <= 0:
        raise ValueError("invalid DeltaNet tensor geometry")
    if name.endswith(("linear_attn.in_proj_qkv.weight", "linear_attn.conv1d.weight")):
        prefix, width = 2 * nk * dk, dv
    elif name.endswith("linear_attn.in_proj_z.weight"):
        prefix, width = 0, dv
    elif name.endswith(("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
                        "linear_attn.A_log", "linear_attn.dt_bias")):
        prefix, width = 0, 1
    else:
        raise ValueError(f"unsupported DeltaNet tensor {name}")
    if x.shape[0] != prefix + nv * width:
        raise ValueError(f"{name}: incompatible leading dimension {x.shape[0]}")
    v = x[prefix:].reshape(nv, width, *x.shape[1:])[idx].reshape(x[prefix:].shape)
    return np.concatenate((x[:prefix], v), axis=0) if prefix else v


class GroupedGDNWeights(Mapping):
    """Experimental load-time row conversion; never selected implicitly by a loader.

    The source must remain unmodified. The returned Spec must accompany these
    weights: mixing it with the original tiled weights changes the model.
    This wrapper does not validate an entire checkpoint or remove its fail-closed gate.
    """
    def __init__(self, source, spec):
        if source.g.kv.get("prism.hadamard.gdn_v_grouped") is not True:
            raise ValueError("grouped conversion requires gdn_v_grouped metadata")
        if not spec.lin_tiled:
            raise ValueError("grouped conversion requires a tiled source spec")
        grouped_indices(spec.lin_nk, spec.lin_heads)
        if spec.lin_dk <= 0 or spec.lin_dv <= 0:
            raise ValueError("invalid DeltaNet head dimensions")
        self.source = source
        self.spec = replace(spec, lin_tiled=False)
        self.nk, self.nv = spec.lin_nk, spec.lin_heads
        self.dk, self.dv = spec.lin_dk, spec.lin_dv

    def __getitem__(self, name):
        value = self.source[name]
        if any(name.endswith(s) for s in _VALUE_INPUTS):
            return reorder_gdn_tensor(name, value, self.nk, self.nv, self.dk, self.dv)
        return value

    def __iter__(self):
        return iter(self.source)

    def __len__(self):
        return len(self.source)

    def signs_for(self, width):
        return self.source.signs_for(width)

    def embedding_table(self):
        return self.source.embedding_table()
