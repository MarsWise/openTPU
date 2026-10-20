# vLLM integration status

## Status (2026-10-20): native vLLM 0.32.0 platform + worker — SOFTWARE VERIFIED (ISA simulator); card NOT RUN

`opentpu/vllm/` is an out-of-tree vLLM platform. vLLM's own engine (its scheduler, continuous
batching, request lifecycle, OpenAI server, streaming) drives `OpenTPUWorker`, which runs every
token of the model on openTPU's engine (`opentpu.llm.qwen3.Engine`) on the ISA simulator or the
card. This is separate from `otpu-serve` (openTPU's own OpenAI-compatible server).

| Part | Where | What it does |
|---|---|---|
| entry point | `pyproject.toml` `vllm.platform_plugins: opentpu = opentpu.vllm:platform_plugin` | active only with `OTPU_VLLM_BACKEND=isa|board|board-sim`; never beside a GPU by accident |
| platform | `opentpu/vllm/platform.py` `OpenTPUPlatform` | `PlatformEnum.OOT`, `device_name="opentpu"`, host tensors on CPU (no CUDA identity); sets `worker_cls`, TP=PP=DP=1, eager, no async scheduling, block size 16, prefix caching off; refuses spec decode / LoRA / vLLM quantization; `validate_request` refuses whole-vocabulary logprobs (-1), prompt logprobs, penalties, logit bias, allowed/bad words, min_tokens, structured outputs (HTTP 400) |
| worker | `opentpu/vllm/worker.py` `OpenTPUWorker(WorkerBase)` | `load_model` builds the Engine with `slots` KV slots (Qwen3, Llama-likes; BitNet with one slot); a request owns a slot while scheduled; prompt chunks run as prefill runs of its slot, all one-token requests as one batched `run_rows` over the slots; host sampling (greedy / temperature / top-k / top-p / min-p, seeded per request) and vLLM's sample logprobs (raw log-softmax of the device's logits, top-k + the sampled token's rank); recompute on preemption; abort frees the slot |
| KV accounting | `get_kv_cache_spec` / `determine_available_memory` | one `FullAttentionSpec` (int8, the model's KV heads and head dim) per attention layer; vLLM's block pool = slots × ⌈max_model_len/16⌉ + 1 null block, so vLLM never admits more than the slots hold. vLLM's block ids are accounting only: the device cache of a request is its slot's contiguous positions |
| BitNet | `opentpu/vllm/__init__.py` `register_models` (entry point `vllm.general_plugins`), `opentpu/vllm/models.py` | vLLM 0.32.0 has no BitNet: `BitNetForCausalLM` is registered as a stand-in class vLLM only inspects (instantiating it raises) and the checkpoint's `quantization_config` `bitnet` as a stand-in QuantizationConfig (offline autobitlinear only); the worker loads the packed ternary checkpoint into the TMM image |
| smoke tool | `tools/vllm_smoke.py` | a checkpoint through `vllm.LLM` on a backend, tokens vs the Engine on the same backend |
| deps | `tools/vllm_requirements.txt` | vLLM's pure-Python deps (the wheel is installed `--no-deps`: it pins CUDA torch 2.13.0) |

Evidence (`tests/test_vllm_plugin.py`, vLLM 0.32.0, torch 2.14.1 CPU in the container — vLLM pins
2.13.0, which the container's index does not serve; transformers 5.19.0; ISA simulator; 8 passed):

- greedy tokens through vLLM == a fresh Engine's for five requests on two slots (they wait for a
  slot), decode rows batched (decode_rows > decode_runs);
- a seeded sampled request gives the same tokens alone and among greedy / other sampled ones;
- preemption (num_gpu_blocks_override=17): a request is preempted and recomputed, tokens unchanged;
- abort frees the slots (at vLLM's next step); a prompt past max_model_len is refused by vLLM;
  a presence penalty is refused;
- `logprobs=3`: the sampled token's and the top 3 logprobs equal the log-softmax of the Engine's
  logits at each step (1e-5), rank 1 for the greedy token;
- `vllm serve` (engine core in its own process): a streamed `/v1/completions` gives the Engine's
  greedy tokens; `presence_penalty` → HTTP 400 and the server goes on;
- a tiny BitNet b1.58 checkpoint (real layout: packed U8 ternary) through `vllm.LLM`: greedy
  tokens == the Engine's TMM image.

Not yet: prompt logprobs, penalties, structured outputs, prefix
caching (the device has slot-to-slot page copies, `serve/kvio.py`: not wired to vLLM's block
hashes), device-side sampling (`llm/batch.py` Steps: reads one word per request instead of the
logits), LFM2 / Qwen3.5 / Gemma 4 / MoE / MiniCPM-V (refused with a message), multimodal inputs,
speculative decoding (MTP stays in otpu-serve). Card run: prepared in the host run script
(`tools/vllm_smoke.py` with `OTPU_VLLM_BACKEND=board`), NOT RUN.

## Earlier status (before 2026-10-20)

The repository includes `otpu-serve` (`opentpu/serve/`, `docs/serve.md`), but an OpenAI-compatible
HTTP endpoint is not a vLLM backend. At that point this branch did not pin or install a vLLM
version.

## Work required for a real backend

1. Pin a vLLM release and inspect its platform/plugin interfaces and version-specific assumptions.
2. Implement model/precision capability negotiation, tokenizer and checkpoint conversion, worker
   and model-runner execution against openTPU.
3. Define block KV allocation/lifetime and device addresses that correspond to actual accelerator
   execution; test isolation, fragmentation, prefix reuse, cancellation and OOM.
4. Exercise GPU-free backend registration, model tokens vs reference, concurrency, streaming and
   shutdown. Report backend status independently of `otpu-serve`.

No user-facing hardware or Vivado work was attempted.

## Design against vLLM 0.32.0 (2026-10-20; implemented as above, with the differences noted there)

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
