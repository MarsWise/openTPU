# Operator coverage matrix

Updated 2026-10-19. One checkpoint graph is covered below (BitNet b1.58 2B4T, hybrid execution).
For CNN, VLM and VLA no checkpoint has been selected, so their rows are a scope/status inventory,
**not** a claim of coverage for a model graph. Existing openTPU RTL and ISA operations remain as documented in `docs/isa.md`.

| Model family | Operator group | Execution location/status in this branch | Test / evidence | Fallback policy |
|---|---|---|---|---|
| Ternary | Packed weight encode/decode | `opentpu/quant.py` (`pack_ternary_mxu`), HF layout in `opentpu/bitnet.py` | `tests/test_ternary.py`, `tests/test_bitnet.py` | — |
| Ternary | Signed ternary matmul | **FPGA**: TMM opcode `0x23` on the MXU (RTL, ISA simulator, compiler lowering) | RTL = ISA bit-exact (`tests/test_rtl.py -k tmm`); board selftest `tmm` stage | — |
| Transformer | GEMM, attention, norm, RoPE, KV, token sampling | Existing supported project paths documented in ISA/model docs; not newly audited for strict offload | No new regression run in this branch | No new fallback guarantees |
| CNN/YOLO | Conv/depthwise/pool/resize/concat/detection | No selected graph or new RTL path | NOT RUN | Unsupported for this branch |
| VLM | Visual encoder/projector/fusion/LLM | No compatible checkpoint selected | NOT RUN | Unsupported for this branch |
| VLA | Observation/state fusion/action head | No authentic pretrained policy selected | NOT RUN | Unsupported for this branch |
| MoE | Router/expert residency/cache scheduling | Existing baseline retained; no changes | No new tests run | Existing documented behavior only |
| vLLM | Platform, worker/model runner, KV allocator | No native vLLM backend implemented | NOT RUN | API exposure is not backend support |

The next coverage increment must be generated from a selected checkpoint's exported graph and
runtime traces. Every host callback and unsupported node must be explicit; `--strict-fpga` has not
been implemented or verified.

## BitNet b1.58 2B4T (hybrid, `opentpu/bitnet_model.py`)

| Operator | Where it runs | Evidence |
|---|---|---|
| q/k/v/o, gate/up/down projections (ternary, `weight_scale`) | **FPGA** TMM (activations: QACT int8 with per-D/4-group power-of-two offsets) | real q_proj bit-exact board = ISA; whole model per BITNET_COMPAT.md |
| Activation quantization for the projections | **FPGA** QACT (differs from HF's per-token absmax int8; close, not identical) | `tests/test_bitnet_model.py` |
| Embedding lookup, tied LM head | Host (numpy) | — |
| RMSNorm, SubLN (`attn_sub_norm`, `ffn_sub_norm`) | Host (numpy) | numpy model = HF transformers (synthetic) |
| RoPE (theta 500000), GQA attention, softmax, KV cache | Host (numpy) | numpy model = HF transformers (synthetic) |
| relu²(gate) · up | Host (numpy) | numpy model = HF transformers (synthetic) |
| Tokenizer | Host (`tokenizers`) | — |

There is no strict mode yet: every host operator above is a documented host operation, not a
silent fallback, but no `--strict-fpga` flag enforces or reports it per run.
