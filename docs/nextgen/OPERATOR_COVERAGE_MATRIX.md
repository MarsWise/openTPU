# Operator coverage matrix

Updated 2026-10-19. One checkpoint graph is covered below (BitNet b1.58 2B4T, hybrid execution).
For CNN, VLM and VLA no checkpoint has been selected, so their rows are a scope/status inventory,
**not** a claim of coverage for a model graph. Existing openTPU RTL and ISA operations remain as documented in `docs/isa.md`.

| Model family | Operator group | Execution location/status in this branch | Test / evidence | Fallback policy |
|---|---|---|---|---|
| Ternary | Packed weight encode/decode | `opentpu/quant.py` (`pack_ternary_mxu`), HF layout in `opentpu/bitnet.py` | `tests/test_ternary.py`, `tests/test_bitnet.py` | — |
| Ternary | Signed ternary matmul | **FPGA**: TMM opcode `0x23` on the MXU (RTL, ISA simulator, compiler lowering) | RTL = ISA bit-exact (`tests/test_rtl.py -k tmm`); board selftest `tmm` stage | — |
| Transformer | GEMM, attention, norm, RoPE, KV, token sampling | Existing supported project paths documented in ISA/model docs; not newly audited for strict offload | No new regression run in this branch | No new fallback guarantees |
| CNN/YOLO | Conv2D (std, 1x1, strided), depthwise, max pool, upsample | **FPGA** kernels (`kernels/conv.py`): conv on the MXU, the rest on the VPU | ISA vs float/torch, RTL = ISA (`tests/test_conv.py`) | YOLOX-Nano graph below |
| VLM | Visual encoder/projector/fusion/LLM | No compatible checkpoint selected | NOT RUN | Unsupported for this branch |
| VLA | SmolVLA: SigLIP tower, connector, VLM prefix, action expert, Euler loop | **FPGA**: every nn.Linear (`kernels/linear.py`), the patch-embedding conv (`kernels/conv.py`) and every attention incl. softmax (`kernels/sdpa.py`, prefix masks) via `opentpu/torchdev.py`; **host**: norms, RoPE, activations, residual adds, pixel shuffle, time embedding, Euler update (elementwise, counted per run) | `tools/smolvla_run.py --tiny`: 100% of MACs on the device, actions rel L2 0.021; `tests/test_torchdev.py` | host ops listed by Coverage; no silent fallback (unsupported convs raise) |
| MoE | Router/expert residency/cache scheduling | Existing baseline retained; no changes | No new tests run | Existing documented behavior only |
| vLLM | Platform, worker, KV accounting | `opentpu/vllm` (vLLM 0.32.0): every token on openTPU's Engine (Qwen3, Llama-likes, BitNet); host: sampling only | `tests/test_vllm_plugin.py` 7 passed (ISA) | unsupported features refused (HTTP 400 / load error) |

The next coverage increment must be generated from a selected checkpoint's exported graph and
runtime traces. Every host callback and unsupported node must be explicit. `--strict-fpga` exists for
the CNN graph (`tools/yolox_graph.py`: unsupported nodes refused, host work limited to the input
write, readout and NMS) and for the LLM engines (`otpu-chat --strict-fpga`,
`opentpu/llm/placement.py`: every stage reported as device / host-move (stored bytes only: a
host-kept table's rows, streamed expert weights) / host; host computation — per-position inputs
without the image's tables, token selection the device loop does not take — is refused;
`tests/test_placement.py`). Not yet for the VLA path (its host attention is reported by
`opentpu/torchdev.py` Coverage instead).

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

## YOLOX-Nano 416² (`tools/yolox_run.py`, ISA simulator, 2026-10-20)

| Operator (count in the graph) | Where it runs | MACs |
|---|---|---|
| Conv2d 1x1 incl. the head's predictions (82), Conv2d 3x3 stem (1) | **FPGA** MXU, `conv2d` (int8 MX, BN folded, SiLU on the VPU) | 503.8 M |
| Depthwise 3x3 (30) | **FPGA** VPU, `depthwise2d` (fp32) | 18.7 M |
| MaxPool 5/9/13 (SPP, 3) | **FPGA** VPU, `maxpool2d` | — |
| Upsample x2 nearest (2) | **FPGA** VPU, `upsample2x` | — |
| Concat (21) | Host (torch.cat) — on the device it is stores into channel slices, not yet wired | — |
| Residual add (8) | Host — a VPU ADD, not yet wired | — |
| Focus (space-to-depth of the input) | Host (input layout) | — |
| Head decode: sigmoid, exp, grid, stride (≈30 small ops), NMS | Host | — |

100% of the model's MACs run on the device kernels; the host keeps memory-layout glue and the
decode. There is no `--strict-fpga` yet: the run prints these host operations per run.

### Whole graph on the device (`opentpu/cnngraph.py`, `tools/yolox_graph.py`)

The same model planned as 123 device ops over NHWC buffers resident in DRAM: concatenations are
producers writing into channel slices (5 copies where a tensor is also read elsewhere), the 8
residual adds are the convolutions' epilogues, the head's sigmoids their activations, and Focus
+ the stem convolution one 6x6 / stride-2 convolution (exact). Left on the host: writing the
input image, reading the three head levels, the box decode (grid, exp, stride) and NMS.
ISA simulator, 416² dog.jpg: 470,154 instructions; detections IoU 0.963 / 0.988 / 0.916 to the
float model's.
