# vLLM integration status

## Status: not implemented

The repository includes `otpu-serve` (`opentpu/serve/`, `docs/serve.md`), but an OpenAI-compatible
HTTP endpoint is not a vLLM backend. This branch did not pin or install a vLLM version and does not
provide an out-of-tree platform, worker, model runner, device allocator, scheduler/KV contract, or
vLLM integration test.

No code fabricates a CUDA device or delegates model execution silently to CPU. No claim is made
that `otpu-serve` implements vLLM's continuous batching or paged KV semantics.

## Work required for a real backend

1. Pin a vLLM release and inspect its platform/plugin interfaces and version-specific assumptions.
2. Implement model/precision capability negotiation, tokenizer and checkpoint conversion, worker
   and model-runner execution against openTPU.
3. Define block KV allocation/lifetime and device addresses that correspond to actual accelerator
   execution; test isolation, fragmentation, prefix reuse, cancellation and OOM.
4. Exercise GPU-free backend registration, model tokens vs reference, concurrency, streaming and
   shutdown. Report backend status independently of `otpu-serve`.

No user-facing hardware or Vivado work was attempted.

## Design against vLLM 0.32.0 (2026-10-20; not implemented)

Pinned for the design: **vLLM 0.32.0** (latest on PyPI on 2026-10-20; the wheel's sources were
read, not installed). Reference out-of-tree backend: the Tenstorrent plugin (vLLM blog,
2026-09-07, built against 0.26.0), the same pattern.

**Entry points** (`vllm/plugins/__init__.py`): `vllm.platform_plugins` -> `opentpu =
opentpu.vllm:platform_plugin`, returning `"opentpu.vllm.platform.OpenTPUPlatform"` only when a
card (`/dev/xdma0`, its firmware contract) or `OTPU_VLLM_BACKEND=isa` is present, so it never
activates beside a GPU; `vllm.general_plugins` -> the model registry (architectures the engine
runs: Qwen3, Llama-likes, LFM2, Qwen3.5, Gemma 4, BitNet).

**Platform** (`vllm/platforms/interface.py`): `_enum = PlatformEnum.OOT`, `device_name =
"opentpu"`, `device_type = "cpu"` and `dispatch_key = "CPU"` for vLLM's own host tensors (no CUDA
identity is claimed); `check_and_update_config(vllm_config)`: `parallel_config.worker_cls =
"opentpu.vllm.worker.OpenTPUWorker"`, eager mode, TP = PP = 1, `cache_config.block_size` =
the engine's attention block, prefix caching and chunked prefill off at first, `max_num_seqs` <=
the image's batch slots, `max_model_len` <= its KV capacity; unsupported settings rejected there.

**Worker** (`vllm/v1/worker/worker_base.py` WorkerBase): `init_device` opens the card;
`load_model` builds `opentpu.llm` Spec + weights and an `Engine(batch=max_num_seqs,
cap=max_model_len, backend=BoardBackend)`; `get_kv_cache_spec` reports the engine's int8 cache
(block size, KV heads, head dim) so the scheduler's block accounting matches the device's;
`initialize_from_config` maps vLLM's KV blocks onto the engine's per-sequence caches (a request
owns a batch slot; its blocks are a slot's contiguous positions); `execute_model(scheduler_output)`
runs the step's phase -- new requests' prompts with `Engine.prefill(seq=slot)`, decodes with
`Engine.step_batch` -- and returns the logits; `sample_tokens` samples on the host (vLLM's
sampler) or on the device (the generate loop's sampler, opentpu/llm/generate.py).

**Limits at first** (as the TT plugin's): phase-homogeneous steps, no speculative decoding /
LoRA / prompt logprobs, preemption by recompute, one card.

**Acceptance:** `vllm serve <model> --device opentpu` (or the platform auto-detected) answers
the OpenAI API with tokens equal to `Engine.generate`'s for the same prompts (greedy), with
continuous batching across requests, cancellation and an out-of-cache error surfaced; tested on
the ISA backend in CI and on the card.
