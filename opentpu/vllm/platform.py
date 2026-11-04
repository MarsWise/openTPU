"""The vLLM platform (vllm.platforms.interface.Platform, vLLM 0.32.0) for openTPU.

check_and_update_config routes every worker to opentpu.vllm.worker.OpenTPUWorker and pins the
settings the engine supports; validate_request refuses, per request, the sampling features the
worker does not implement (rather than ignoring them)."""
from __future__ import annotations

import contextlib
import os

import torch
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.platforms.interface import Platform, PlatformEnum

from . import ENV_BACKEND, VLLM_VERSION

logger = init_logger(__name__)

WORKER = "opentpu.vllm.worker.OpenTPUWorker"
BLOCK_SIZE = 16             # vLLM's KV accounting block (the device's caches: one slot each)


class Unsupported(VLLMValidationError):
    """A configuration or request feature the openTPU backend does not implement (a client
    error: the OpenAI server answers 400 with this message)."""


class OpenTPUPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name = "opentpu"
    device_type = "cpu"         # vLLM's host-side tensors; the model runs on the openTPU backend
    dispatch_key = "CPU"
    dist_backend = "gloo"
    supported_quantization: list[str] = ["bitnet"]  # BitNet's packed ternary checkpoint
    #                                              (the engine reads it); otherwise the engine
    #                                              quantizes the float checkpoint itself

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return f"openTPU ({os.environ.get(ENV_BACKEND, '?')})"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        from .worker import card_dram_bytes
        return card_dram_bytes()

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        pass

    @classmethod
    def get_current_memory_usage(cls, device=None) -> float:
        return 0.0

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        if quant == "bitnet":
            return
        raise Unsupported(f"vLLM quantization {quant!r}: the openTPU engine quantizes the "
                          "checkpoint itself (load it unquantized, or BitNet's offline format)")

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return False

    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        import vllm
        if vllm.__version__.split("+")[0] != VLLM_VERSION:
            logger.warning("openTPU's vLLM platform was written against vLLM %s, running %s",
                           VLLM_VERSION, vllm.__version__)
        pc, sc = vllm_config.parallel_config, vllm_config.scheduler_config
        cc, mc = vllm_config.cache_config, vllm_config.model_config
        if pc.tensor_parallel_size != 1 or pc.pipeline_parallel_size != 1 \
                or getattr(pc, "data_parallel_size", 1) != 1:
            raise Unsupported("openTPU: one card (tensor / pipeline / data parallel size 1)")
        if vllm_config.speculative_config is not None:
            raise Unsupported("openTPU: no speculative decoding through vLLM (the engine's MTP "
                              "runs in otpu-serve)")
        if vllm_config.lora_config is not None:
            raise Unsupported("openTPU: no LoRA")
        if pc.worker_cls in ("auto", None, ""):
            pc.worker_cls = WORKER
        elif pc.worker_cls != WORKER:
            raise Unsupported(f"openTPU: worker {pc.worker_cls} (needs {WORKER})")
        if pc.distributed_executor_backend in (None, "auto"):
            pc.distributed_executor_backend = "uni"
        sc.async_scheduling = False
        cc.block_size = BLOCK_SIZE
        cc.user_specified_block_size = True
        if cc.enable_prefix_caching:
            logger.info("openTPU: prefix caching off (each request owns a device KV slot)")
            cc.enable_prefix_caching = False
        if mc is not None:
            mc.enforce_eager = True
        comp = vllm_config.compilation_config
        with contextlib.suppress(Exception):
            from vllm.config import CompilationMode, CUDAGraphMode
            comp.mode = CompilationMode.NONE
            comp.cudagraph_mode = CUDAGraphMode.NONE

    @classmethod
    def validate_request(cls, processed_inputs, params) -> None:
        from vllm.sampling_params import SamplingParams
        if not isinstance(params, SamplingParams):
            raise Unsupported("openTPU: generation only (no pooling)")
        bad = []
        if params.logprobs is not None and params.logprobs < 0:
            bad.append("logprobs over the whole vocabulary (-1)")
        if params.prompt_logprobs is not None:
            bad.append("prompt_logprobs")
        if params.presence_penalty or params.frequency_penalty:
            bad.append("presence / frequency penalty")
        if params.repetition_penalty not in (None, 1.0):
            bad.append("repetition_penalty")
        if params.logit_bias:
            bad.append("logit_bias")
        if params.allowed_token_ids:
            bad.append("allowed_token_ids")
        if params.bad_words:
            bad.append("bad_words")
        if params.min_tokens:
            bad.append("min_tokens")
        if getattr(params, "structured_outputs", None) is not None:
            bad.append("structured outputs")
        if bad:
            raise Unsupported("openTPU's vLLM worker does not implement: " + ", ".join(bad))
