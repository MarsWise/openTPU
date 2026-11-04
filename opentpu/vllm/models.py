"""Stand-in model classes for architectures vLLM does not implement but openTPU's engine runs
(BitNet b1.58: ternary TMM). vLLM resolves a checkpoint's architecture to a class and inspects
its interfaces; with the openTPU platform its worker never builds that class (the worker loads
the checkpoint into openTPU's engine), so instantiating one is an error, not a CPU fallback."""
from __future__ import annotations

import torch
from torch import nn


class OpenTPUOnlyForCausalLM(nn.Module):
    def __init__(self, vllm_config=None, prefix: str = ""):
        super().__init__()
        raise RuntimeError("this architecture runs only in openTPU's engine "
                           "(OTPU_VLLM_BACKEND with the openTPU platform)")

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("not executable outside openTPU's engine")

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, intermediate_tensors=None,
                inputs_embeds=None):
        raise RuntimeError("not executable outside openTPU's engine")

    def compute_logits(self, hidden_states: torch.Tensor):
        raise RuntimeError("not executable outside openTPU's engine")


def bitnet_quant_config():
    """BitNet b1.58's checkpoint quantization ("bitnet", offline autobitlinear: packed ternary
    weights and their scales), named so vLLM accepts the checkpoint's quantization_config; the
    openTPU worker reads the packed weights itself (opentpu/llm/bitnet.py) and runs them as
    TMM's ternary rows. vLLM never quantizes or computes a layer with it."""
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

    class BitNetOpenTPU(QuantizationConfig):
        def __init__(self, cfg: dict | None = None):
            super().__init__()
            self.cfg = cfg or {}

        def get_name(self):
            return "bitnet"

        def get_supported_act_dtypes(self):
            return [torch.float32, torch.bfloat16, torch.float16]

        @classmethod
        def get_min_capability(cls) -> int:
            return 0

        @staticmethod
        def get_config_filenames() -> list[str]:
            return []

        @classmethod
        def from_config(cls, config: dict):
            if config.get("linear_class", "autobitlinear") != "autobitlinear" \
                    or config.get("quantization_mode", "offline") != "offline":
                raise ValueError("openTPU runs BitNet's offline autobitlinear checkpoints only")
            return cls(config)

        def get_quant_method(self, layer, prefix: str):
            raise RuntimeError("BitNet's layers run only in openTPU's engine")
    return BitNetOpenTPU
