"""vLLM's worker on openTPU (vllm.v1.worker.worker_base.WorkerBase, vLLM 0.32.0).

One process, one card (or the ISA simulator). load_model builds an opentpu.llm Engine for the
checkpoint vLLM was given -- the same image otpu-serve runs -- with `slots` KV slots: each vLLM
request owns a slot while it is scheduled (its device KV cache: the slot's contiguous
positions). vLLM's scheduler does the continuous batching; its KV blocks are the accounting of
those slots (get_kv_cache_spec reports the device's int8 cache per attention layer, so vLLM's
block budget equals the slots' tokens) and its block ids are not addresses on the card.

execute_model runs the step vLLM scheduled: each request's prompt chunk in prefill runs of its
slot (Engine.prefill_chunks), and every request with one token to feed in one batched run
(Engine.run_rows over their slots). sample_tokens samples each request whose scheduled tokens
reach its last known token from the logits the device computed (greedy, temperature, top-k,
top-p, min-p; seeded per request; vLLM's sample logprobs). A request that is preempted is recomputed from its tokens
(vLLM's recompute preemption). Unsupported features are refused (platform.validate_request,
check_and_update_config, Unsupported here) -- nothing runs on the CPU in the model's place.
"""
from __future__ import annotations

import dataclasses
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.outputs import LogprobsLists, ModelRunnerOutput
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

from . import ENV_BACKEND
from .platform import BLOCK_SIZE, Unsupported

logger = init_logger(__name__)

DEFAULTS = dict(dev="/dev/xdma0", slots=4, wformat="auto", head_format=None,
                resident=True, prompt_runs=True)


def card_dram_bytes() -> int:
    """The device memory vLLM's platform reports (informational: the KV budget is the slots')."""
    return int(float(os.environ.get("OTPU_VLLM_DRAM_GB", "2")) * 2**30)


def options(vllm_config) -> dict:
    """The worker's options: DEFAULTS, then vLLM's additional_config["opentpu"], then the
    environment (OTPU_VLLM_DEV, OTPU_VLLM_SLOTS)."""
    o = dict(DEFAULTS, backend=os.environ.get(ENV_BACKEND, "isa"))
    o.update((getattr(vllm_config, "additional_config", None) or {}).get("opentpu", {}))
    if os.environ.get("OTPU_VLLM_DEV"):
        o["dev"] = os.environ["OTPU_VLLM_DEV"]
    if os.environ.get("OTPU_VLLM_SLOTS"):
        o["slots"] = int(os.environ["OTPU_VLLM_SLOTS"])
    return o


def build_engine(path, backend: str, cap: int, slots: int, dev: str = "/dev/xdma0",
                 wformat: str = "auto", head_format=None, resident: bool = True,
                 prompt_runs: bool = True, cfg=None):
    """(engine, spec) for a checkpoint directory as otpu-serve builds it (opentpu/serve/api.py
    load): Qwen3 and the Llama-likes with `slots` KV slots, BitNet b1.58 (ternary TMM image)."""
    from ..host import chat as CH
    from ..llm import formats as FM
    from ..llm import load_spec, qwen3
    from ..llm.qwen3 import PREFILL_ROWS, Engine, device_config
    path = Path(path)
    spec = load_spec(path)
    if type(spec) is not qwen3.Spec:
        raise Unsupported(f"{path.name}: {type(spec).__module__} models are not in the vLLM "
                          "worker yet (Qwen3, Llama-likes and BitNet are); use otpu-serve")
    bitnet = getattr(spec, "subln", False)
    if bitnet:
        from ..llm.bitnet import IMAGE, Weights
        W, kw = Weights(path), dict(IMAGE)
        if head_format:                     # (the LM head; the projections stay ternary)
            kw["head_format"] = head_format
        # slots: one slice (its MLP's sub-norm spans the whole row), any number of sequences:
        # their decode rows go through one run (run_rows), each TMM over all of them (M > 1)
        if backend == "isa" and cfg is None:
            from ..isasim import board_config
            cfg = board_config(DRAM_BYTES=1 << 28)
    else:
        W = qwen3.load_weights(path)
        kw = dict(wformat=FM.auto(spec, wformat), head_format=head_format)
    B = max(1, int(slots))
    be, ccfg = CH.make_backend(backend, spec, cap, dev, path.name, lookup=True,
                               wformat=kw["wformat"], head_format=kw.get("head_format"))
    cfg = cfg or ccfg
    if cfg is None:                         # the simulator: room for the slots, one slice
        c = device_config(spec, cap, batch=B, rows=PREFILL_ROWS, lookup=True, S=1,
                          wformat=kw["wformat"], head_format=kw.get("head_format"))
        cfg = dataclasses.replace(c, DRAM_BYTES=2 * c.DRAM_BYTES)
    eng = Engine(spec, W, cap=cap, cfg=cfg, backend=be, resident=resident,
                 prompt_runs=prompt_runs, batch=B, **kw)
    return eng, spec


@dataclasses.dataclass
class _Req:
    rid: str
    slot: int
    tokens: list            # prompt + sampled tokens (the device's slot holds tokens[:pos])
    params: object
    rng: np.random.Generator


def sample(logits: np.ndarray, params, rng: np.random.Generator) -> int:
    """One token from a row of logits under vLLM's SamplingParams (the subset the platform
    accepts): greedy at temperature 0, else temperature, top-k, top-p, min-p."""
    t = float(params.temperature or 0.0)
    if t < 1e-5:
        return int(np.argmax(logits))
    x = logits.astype(np.float64) / t
    k = int(params.top_k or 0)
    if 0 < k < x.size:
        kth = np.partition(x, -k)[-k]
        x = np.where(x < kth, -np.inf, x)
    p = np.exp(x - x.max())
    p /= p.sum()
    mp = float(getattr(params, "min_p", 0.0) or 0.0)
    if mp > 0:
        p = np.where(p < mp * p.max(), 0.0, p)
        p /= p.sum()
    tp = float(params.top_p if params.top_p is not None else 1.0)
    if tp < 1.0:
        order = np.argsort(-p, kind="stable")
        c = np.cumsum(p[order])
        keep = order[:int(np.searchsorted(c, tp) + 1)]
        q = np.zeros_like(p)
        q[keep] = p[keep]
        p = q / q.sum()
    return int(rng.choice(p.size, p=p))


def _logprobs(ids, lps) -> LogprobsLists | None:
    """vLLM's sample logprobs (its default logprobs_mode, raw_logprobs: log-softmax of the
    device's logits before any temperature): per request row the sampled token first, then the
    top k; the sampled token's rank among all (1 = the most likely)."""
    if not lps:
        return None
    K = max(k for _, _, k in lps.values())
    tok = np.zeros((len(ids), K + 1), np.int64)
    val = np.zeros((len(ids), K + 1), np.float32)
    rank = np.zeros(len(ids), np.int64)
    for i, rid in enumerate(ids):
        if rid not in lps:
            continue
        row, t, k = lps[rid]
        x = row.astype(np.float64)
        lp = x - (x.max() + np.log(np.exp(x - x.max()).sum()))
        top = np.argsort(-lp, kind="stable")[:K]
        tok[i, 0], val[i, 0] = t, lp[t]
        tok[i, 1:], val[i, 1:] = top, lp[top]
        rank[i] = int((lp > lp[t]).sum()) + 1
    return LogprobsLists(tok, val, rank)


class OpenTPUWorker(WorkerBase):
    def __init__(self, vllm_config, local_rank: int, rank: int, distributed_init_method: str,
                 is_driver_worker: bool = False, **kwargs):
        super().__init__(vllm_config, local_rank, rank, distributed_init_method,
                         is_driver_worker)
        self.opts = options(vllm_config)
        self.eng = None
        self.spec = None
        self.reqs: dict[str, _Req] = {}
        self.kept: dict[str, tuple] = {}    # a preempted request's sampling (params, rng)
        self.free: list[int] = []
        self._pending = None
        self.resumed = 0
        self.stats = dict(steps=0, prefill_runs=0, decode_runs=0, decode_rows=0, device_s=0.0)

    # ---- set-up
    def init_device(self) -> None:
        self.device = torch.device("cpu")
        dev = self.opts["dev"]          # an XDMA prefix: its nodes are <dev>_user, <dev>_h2c_0 ...
        if self.opts["backend"] == "board" and not (os.path.exists(dev) or
                                                    os.path.exists(f"{dev}_h2c_0")):
            raise RuntimeError(f"openTPU: {dev}_h2c_0 not found (the card's XDMA device)")

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        if load_dummy_weights:
            raise Unsupported("openTPU: dummy weights (load_format=dummy)")
        mc = self.model_config
        self.cap = int(math.ceil(mc.max_model_len / 128) * 128)
        t0 = time.perf_counter()
        self.eng, self.spec = build_engine(mc.model, self.opts["backend"], self.cap,
                                           self.opts["slots"], self.opts["dev"],
                                           self.opts["wformat"], self.opts["head_format"],
                                           self.opts["resident"], self.opts["prompt_runs"],
                                           self.opts.get("cfg"))
        self.B = self.eng.batch
        self.free = list(range(self.B))[::-1]
        logger.info("openTPU: %s on %s, %d KV slot(s) of %d positions, image %.0f MiB, %.1f s",
                    Path(mc.model).name, self.opts["backend"], self.B, self.cap,
                    self.eng.image.nbytes / 2**20, time.perf_counter() - t0)
        sc = self.scheduler_config
        if sc.max_num_seqs > self.B:
            logger.info("openTPU: max_num_seqs %d -> %d (the KV slots)", sc.max_num_seqs, self.B)
            sc.max_num_seqs = self.B

    def get_model(self):
        return None

    def get_supported_tasks(self):
        return ("generate",)

    def compile_or_warm_up_model(self) -> CompilationTimes:
        return CompilationTimes(language_model=0.0, encoder=0.0)

    def check_health(self) -> None:
        return

    def shutdown(self) -> None:
        """Releases the card now (its lock with it): vLLM's in-process engine core can keep the
        worker referenced after shutdown, and a later user of the card in the same process
        (tools/vllm_smoke.py's reference engine) then met DeviceBusy (r4 and r5-m8 runs)."""
        eng, self.eng = self.eng, None
        close = getattr(getattr(eng, "backend", None), "close", None)
        if callable(close):
            close()

    # ---- the KV accounting: the slots' positions as vLLM blocks
    def _attn_layers(self) -> list[str]:
        return [f"model.layers.{i}.self_attn.attn" for i in range(self.spec.layers)]

    def get_kv_cache_spec(self):
        s = self.spec
        one = FullAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=s.n_kv,
                                head_size=s.head_dim, dtype=torch.int8)
        return {n: one for n in self._attn_layers()}

    def get_supported_kv_cache_layouts(self) -> list[str]:
        # (the device's caches have their own layout, kvio.Layout; vLLM allocates no tensors)
        return ["LBNHC"]

    def set_kv_cache_layout(self, kv_cache_layout: str) -> None:
        self.kv_layout = kv_cache_layout

    def blocks_needed(self) -> int:
        """The slots' positions in vLLM blocks, plus vLLM's null block."""
        per = math.ceil(self.model_config.max_model_len / BLOCK_SIZE)
        return self.B * per + 1

    def determine_available_memory(self) -> int:
        page = next(iter(self.get_kv_cache_spec().values())).page_size_bytes
        return self.blocks_needed() * page * self.spec.layers

    def get_cache_block_size_bytes(self) -> int:
        page = next(iter(self.get_kv_cache_spec().values())).page_size_bytes
        return page * self.spec.layers

    def initialize_from_config(self, kv_cache_config) -> None:
        self.kv_blocks = kv_cache_config.num_blocks
        logger.info("openTPU: vLLM KV blocks %d (slots need %d)", self.kv_blocks,
                    self.blocks_needed())

    def update_max_model_len(self, n: int) -> None:
        if n > self.cap:
            raise Unsupported(f"openTPU: max_model_len {n} past the slots' {self.cap}")
        self.model_config.max_model_len = n

    # ---- the step
    def _take_slot(self, rid: str, tokens, params, seed) -> _Req:
        if not self.free:
            raise RuntimeError(f"openTPU: no free KV slot for {rid} ({self.B} slots)")
        s = self.free.pop()
        self.eng.reset(seq=s)
        if rid in self.kept:
            params, rng = self.kept.pop(rid)
        else:
            rng = np.random.default_rng(seed)
        r = _Req(rid, s, list(tokens), params, rng)
        self.reqs[rid] = r
        return r

    def _drop(self, rid: str, keep: bool = False) -> None:
        r = self.reqs.pop(rid, None)
        if r is not None:
            self.free.append(r.slot)
            if keep:
                self.kept[rid] = (r.params, r.rng)
        if not keep:
            self.kept.pop(rid, None)

    def execute_model(self, scheduler_output):
        so, eng = scheduler_output, self.eng
        for rid in so.finished_req_ids:
            self._drop(rid)
        for rid in so.preempted_req_ids or ():
            self._drop(rid, keep=True)
        seed0 = self.model_config.seed or 0
        for nr in so.scheduled_new_reqs:
            if nr.prompt_token_ids is None:
                raise Unsupported("openTPU: prompts as token ids or text (no prompt embeds)")
            if nr.mm_features:
                raise Unsupported("openTPU: no multimodal inputs through vLLM yet")
            if nr.num_computed_tokens:
                raise RuntimeError(f"openTPU: {nr.req_id} starts at {nr.num_computed_tokens} "
                                   "(prefix caching is off)")
            p = nr.sampling_params
            self._take_slot(nr.req_id, nr.prompt_token_ids, p,
                            p.seed if p.seed is not None else (seed0, len(self.reqs)))
        cr = so.scheduled_cached_reqs
        for i, rid in enumerate(cr.req_ids):
            if rid in cr.resumed_req_ids:           # recompute after preemption
                self._drop(rid, keep=True)
                self._take_slot(rid, cr.all_token_ids[rid], None, None)
                self.resumed += 1
            r = self.reqs[rid]
            if eng.poss[r.slot] != cr.num_computed_tokens[i]:
                raise RuntimeError(f"openTPU: {rid} at {eng.poss[r.slot]} on the device, "
                                   f"{cr.num_computed_tokens[i]} in vLLM")
        logits, decode = {}, []
        t0 = time.perf_counter()
        for rid, n in so.num_scheduled_tokens.items():
            r = self.reqs[rid]
            p0 = eng.poss[r.slot]
            part = r.tokens[p0:p0 + n]
            if len(part) != n:
                raise RuntimeError(f"openTPU: {rid} scheduled {n} tokens, {len(part)} known")
            last = p0 + n == len(r.tokens)
            if n == 1 and last and self.B > 1:
                decode.append(r)
                continue
            lg = None
            for _, lg in eng.prefill_chunks(part, seq=r.slot, logits=last):
                self.stats["prefill_runs"] += 1
            if last:
                logits[rid] = lg
        for i in range(0, len(decode), self.B):
            grp = decode[i:i + self.B]
            lg = eng.run_rows([(r.slot, eng.poss[r.slot]) for r in grp],
                              [r.tokens[-1] for r in grp], list(range(len(grp))))
            for r, row in zip(grp, lg):
                eng.poss[r.slot] += 1
                logits[r.rid] = row
            self.stats["decode_runs"] += 1
            self.stats["decode_rows"] += len(grp)
        self.stats["device_s"] += time.perf_counter() - t0
        self.stats["steps"] += 1
        self._pending = (list(so.num_scheduled_tokens), logits)
        return None

    def sample_tokens(self, grammar_output) -> ModelRunnerOutput:
        if grammar_output is not None:
            raise Unsupported("openTPU: no structured outputs")
        ids, logits = self._pending
        self._pending = None
        out, lps = [], {}
        for rid in ids:
            if rid in logits:
                r = self.reqs[rid]
                row = np.asarray(logits[rid])
                t = sample(row, r.params, r.rng)
                r.tokens.append(t)
                out.append([t])
                if r.params.logprobs is not None:
                    lps[rid] = (row, t, int(r.params.logprobs))
            else:
                out.append([])
        return ModelRunnerOutput(req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)},
                                 sampled_token_ids=out, logprobs=_logprobs(ids, lps))
