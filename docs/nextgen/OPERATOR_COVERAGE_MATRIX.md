# Operator coverage matrix

No end-to-end checkpoint has been selected for the new BitNet/Tern2, CNN, VLM or VLA work.
Therefore this matrix is a per-family scope/status inventory, **not** a claim of coverage for a
model graph. Existing openTPU RTL and ISA operations remain as documented in `docs/isa.md`.

| Model family | Operator group | Execution location/status in this branch | Test / evidence | Fallback policy |
|---|---|---|---|---|
| Ternary reference only | Packed weight encode/decode | Python reference (`opentpu/quant.py`), not RTL | `tests/test_ternary.py` | Not a model execution path |
| Ternary reference only | Signed ternary dot/matmul | Direct packed positive/negative/zero accumulation in Python int64 | 23 isolated pytest cases; scalar golden comparisons | No checkpoint runtime dispatch |
| Transformer | GEMM, attention, norm, RoPE, KV, token sampling | Existing supported project paths documented in ISA/model docs; not newly audited for strict offload | No new regression run in this branch | No new fallback guarantees |
| CNN/YOLO | Conv/depthwise/pool/resize/concat/detection | No selected graph or new RTL path | NOT RUN | Unsupported for this branch |
| VLM | Visual encoder/projector/fusion/LLM | No compatible checkpoint selected | NOT RUN | Unsupported for this branch |
| VLA | Observation/state fusion/action head | No authentic pretrained policy selected | NOT RUN | Unsupported for this branch |
| MoE | Router/expert residency/cache scheduling | Existing baseline retained; no changes | No new tests run | Existing documented behavior only |
| vLLM | Platform, worker/model runner, KV allocator | No native vLLM backend implemented | NOT RUN | API exposure is not backend support |

The next coverage increment must be generated from a selected checkpoint's exported graph and
runtime traces. Every host callback and unsupported node must be explicit; `--strict-fpga` has not
been implemented or verified.
