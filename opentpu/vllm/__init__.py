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


def register_models() -> None:
    """vLLM's general plugin hook: architectures openTPU runs that vLLM itself does not know
    (BitNet b1.58), registered as stand-ins whose class vLLM only inspects -- the openTPU worker
    builds its own engine and never instantiates them (opentpu/vllm/models.py)."""
    if backend_name() is None:
        return
    from vllm import ModelRegistry
    from vllm.model_executor.layers.quantization import (QUANTIZATION_METHODS,
                                                         register_quantization_config)
    if "BitNetForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("BitNetForCausalLM",
                                     "opentpu.vllm.models:OpenTPUOnlyForCausalLM")
    if "bitnet" not in QUANTIZATION_METHODS:
        from .models import bitnet_quant_config
        register_quantization_config("bitnet")(bitnet_quant_config())
