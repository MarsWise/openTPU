"""openTPU as a vLLM out-of-tree platform (pinned: vLLM 0.32.0; docs/nextgen/VLLM_INTEGRATION.md).

Entry point (pyproject.toml): ``vllm.platform_plugins: opentpu = opentpu.vllm:platform_plugin``.
The plugin activates only when asked to -- ``OTPU_VLLM_BACKEND`` set (``isa`` for the ISA
simulator, ``board`` for the card at ``OTPU_VLLM_DEV``, default ``/dev/xdma0``) -- so installing
opentpu beside a GPU never changes vLLM's platform. No CUDA identity is claimed: the platform is
``opentpu``, vLLM's own host tensors live on the CPU, and every token's model computation runs
in openTPU's engine (opentpu.llm.qwen3.Engine) on the selected backend.
"""
from __future__ import annotations

import os

VLLM_VERSION = "0.32.0"         # the release the platform / worker interfaces were written to
ENV_BACKEND = "OTPU_VLLM_BACKEND"
BACKENDS = ("isa", "board", "board-sim")


def backend_name() -> str | None:
    b = os.environ.get(ENV_BACKEND, "").strip()
    return b or None


def platform_plugin() -> str | None:
    """vLLM's platform plugin hook: the platform's qualified name when activated, else None."""
    b = backend_name()
    if b is None:
        return None
    if b not in BACKENDS:
        raise ValueError(f"{ENV_BACKEND}={b!r}: one of {', '.join(BACKENDS)}")
    return "opentpu.vllm.platform.OpenTPUPlatform"
