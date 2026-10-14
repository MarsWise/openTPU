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
