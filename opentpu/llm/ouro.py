"""Ouro checkpoint geometry and native INT4 preparation, not an inference backend.

Ouro shares one layer stack across recurrent steps but has a distinct KV cache
per (step, layer). Its four sandwich norms and final norm after each step are
not implemented by the current generic Qwen engine. Do not route load_spec to
that engine until an actual Ouro executor has been validated.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import json
from pathlib import Path


@dataclass(frozen=True)
class OuroSpec:
    layers: int
    hidden: int
    ffn: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    max_context: int
    theta: float
    eps: float
    steps: int

    @classmethod
    def from_hf(cls, path, steps: int | None = None):
        c = json.loads((Path(path) / 'config.json').read_text(encoding='utf-8'))
        if c.get('model_type') != 'ouro' or c.get('hidden_act') != 'silu':
            raise ValueError('expected an Ouro SiLU checkpoint')
        layers = int(c['num_hidden_layers'])
        if c.get('layer_types') != ['full_attention'] * layers or c.get('use_sliding_window'):
            raise ValueError('only full-attention Ouro layers are supported')
        if c.get('rope_scaling') or c.get('tie_word_embeddings'):
            raise ValueError('unsupported Ouro RoPE or tied embeddings')
        steps = int(c.get('total_ut_steps', 4) if steps is None else steps)
        if steps not in (1, 2, 3, 4):
            raise ValueError('Ouro recurrent steps must be 1..4')
        spec = cls(layers, int(c['hidden_size']), int(c['intermediate_size']),
                   int(c['num_attention_heads']), int(c['num_key_value_heads']),
                   int(c['head_dim']), int(c['vocab_size']), int(c['max_position_embeddings']),
                   float(c['rope_theta']), float(c['rms_norm_eps']), steps)
        if min(spec.layers, spec.hidden, spec.ffn, spec.heads, spec.kv_heads,
               spec.head_dim, spec.vocab, spec.max_context) <= 0 or \
                spec.hidden != spec.heads * spec.head_dim or spec.heads % spec.kv_heads:
            raise ValueError('invalid Ouro dimensions')
        return spec

    def cache_index(self, step: int, layer: int) -> int:
        if not 0 <= step < self.steps or not 0 <= layer < self.layers:
            raise ValueError('invalid Ouro recurrent step or layer')
        return step * self.layers + layer

    def kv_payload_bytes(self, context: int, bytes_per_element: int = 1) -> int:
        """Lower bound: K and V payload only; excludes alignment, scales and metadata."""
        if not 0 < context <= self.max_context or bytes_per_element not in (1, 2, 4):
            raise ValueError('invalid context or KV element size')
        return (self.steps * self.layers * context * 2 * self.kv_heads *
                self.head_dim * bytes_per_element)


@dataclass(frozen=True)
class RecurrentKVLayout:
    """Conservative per-slice KV regions; all steps and layers remain live."""
    spec: OuroSpec
    cfg: object
    cap: int
    padded_dim: int
    local_heads: int
    plan: object

    @cached_property
    def _addresses(self):
        return {a.name: a.address for a in self.plan.allocations}

    def buffer_name(self, step: int, layer: int, local_head: int, kind: str) -> str:
        self.spec.cache_index(step, layer)
        if not 0 <= local_head < self.local_heads or kind not in ('k', 'ks', 'vt', 'vs'):
            raise ValueError('invalid Ouro KV head or component')
        return f'ouro.kv.{step}.{layer}.{local_head}.{kind}'

    def descriptor(self, step: int, layer: int, sid: int):
        from opentpu.compiler import KVDesc
        self.spec.cache_index(step, layer)
        if not 0 <= sid < self.cfg.S:
            raise ValueError('invalid Ouro KV slice')
        heads = {}
        for local, head in enumerate(range(sid, self.spec.kv_heads, self.cfg.S)):
            heads[head] = {kind: self._addresses[self.buffer_name(step, layer, local, kind)]
                           for kind in ('k', 'ks', 'vt', 'vs')}
        return KVDesc(heads, self.cap, self.padded_dim, self.cfg.D, self.cfg.S, sid,
                      dv=self.spec.head_dim)


def plan_recurrent_kv(spec: OuroSpec, cfg, cap: int, *, lifetime_end: int,
                      weights_bytes: int = 0, reserve_bytes: int = 0) -> RecurrentKVLayout:
    """Allocate K, K scales, V-transpose and V scales for each (step, layer).

    The same local-head addresses are reused on independent slices. All buffers
    live over [0, lifetime_end); the graph owner must supply its actual horizon.
    No CPU model computation or device execution occurs here.
    """
    from opentpu.compiler import k_scale_bytes
    from opentpu.nextgen import Allocation, MemoryPlan, MemoryBudgetError
    D, S, d = cfg.D, cfg.S, spec.head_dim
    if (not isinstance(cap, int) or cap <= 0 or cap % 4 or cap > spec.max_context or
            S <= 0 or D <= 0 or D & (D - 1) or d <= 0 or d % 2 or
            (d > D and d % D) or spec.kv_heads <= 0 or spec.layers <= 0 or
            spec.steps not in (1, 2, 3, 4) or not isinstance(lifetime_end, int) or
            isinstance(lifetime_end, bool) or lifetime_end <= 0):
        raise ValueError('invalid Ouro KV geometry')
    dk = -(-d // D) * D
    nlocal = -(-spec.kv_heads // S)
    if weights_bytes < 0 or reserve_bytes < 0:
        raise ValueError('Ouro memory sizes must not be negative')
    alignment = max(128, D)
    cursor = (weights_bytes + alignment - 1) // alignment * alignment
    limit = cfg.DRAM_BYTES - reserve_bytes
    if cursor > limit:
        raise MemoryBudgetError('Ouro weights and reserve exceed device memory')
    allocations = []
    for step in range(spec.steps):
        for layer in range(spec.layers):
            for local in range(nlocal):
                for kind, size in (('k', cap * dk), ('ks', cap * k_scale_bytes(dk, D)),
                                   ('vt', cap * dk), ('vs', cap * 4)):
                    address = cursor
                    cursor = (cursor + size + alignment - 1) // alignment * alignment
                    if cursor > limit:
                        raise MemoryBudgetError('Ouro recurrent KV exceeds device memory')
                    allocations.append(Allocation(f'ouro.kv.{step}.{layer}.{local}.{kind}',
                                                  address, size, 0, lifetime_end))
    plan = MemoryPlan(weights_bytes, cfg.DRAM_BYTES, alignment, tuple(allocations), cursor,
                      reserve_bytes)
    plan.verify()
    return RecurrentKVLayout(spec, cfg, cap, dk, nlocal, plan)
