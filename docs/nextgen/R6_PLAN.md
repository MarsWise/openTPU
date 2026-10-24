# r6: no host computation (plan, 2026-10-24)

Goal: every model the project supports runs with **no computation on the host**. The host
keeps only what is I/O by nature: decoding a file or a camera frame into pixels, the tokenizer
and detokenizer (text <-> ids), writing the input and reading the final result, and choosing
which program runs next (the scheduler). Everything between input and output runs on the card,
from weights and programs resident in its DDR3.

Status labels as elsewhere in docs/nextgen: SOURCE INSPECTION unless marked otherwise. Nothing
here is built; r5 (`0f9f9b1d`, m8 @ 150 MHz) is being implemented by the user.

## 1. Where the host computes today (source inspection of this tree, `0f3876fe`)

| Path | Host computation today | Evidence |
|---|---|---|
| LLM engines (`opentpu/llm/qwen3.py` Engine and the engines on it: Qwen3, Llama-likes, LFM2, Qwen3.5, Gemma 4, BitNet via the engine) | **none** with `generate_card`: embeddings + RoPE from the image's tables, layers, head, ARGMAX / the device sampler (temperature, top-k, top-p, repetition penalty) on the device. Host-move only: MoE experts streamed from a pool file, Gemma 4 PLE rows | `opentpu/llm/placement.py`; `otpu-chat --strict-fpga` on r4 (BitNet, run 4) |
| BitNet hybrid model (`opentpu/bitnet_model.py`) | embedding, norms, RoPE, attention, relu^2 (numpy) | OPERATOR_COVERAGE_MATRIX.md; superseded by the engine path |
| vLLM platform (`opentpu/vllm`) | token sampling (vLLM's sampler on the logits) | OPERATOR_COVERAGE_MATRIX.md |
| CNN graph (`opentpu/cnngraph.py`, YOLOX) | NMS; with `--strict-fpga` the box decode is on the device | `tools/yolox_graph.py`; card r4 run 4e |
| VLM (MiniCPM-V, `opentpu/llm/minicpmv.py`) | image preprocessing: slice grid, bicubic resizes, normalization, 14x14 patches, position ids (as a processor does) | module docstring |
| VLA (SmolVLA via `opentpu/torchdev.py`) | **most**: LayerNorm / RMSNorm, RoPE, GELU / SiLU, residual adds, pow, pixel shuffle, the time embedding (sin/cos), the Euler update; the graph itself is PyTorch, one device call per layer with its weights re-sent every call | run 4 log: 398 M host elements a sample; 5643 calls, 2.35 G card cycles (17.6 s) in 1415 s of calls |

Measured cost of the per-layer pattern (r4, SmolVLA base, PHYSICAL FPGA, timing only — that
run's numerics were void, see PROGRESS.md): 99 % of the time in the device calls is transfer,
program load and synchronisation, not computation.

## 2. What removes it, and whether it needs RTL

### 2a. Software on r5 (no new bitstream)

| Item | How | Units used |
|---|---|---|
| VLA graph on the device | a resident image + a program chain per policy step (as YOLOX's `cnngraph` and the LLM engines): weights loaded once, activations in TMEM / DDR3, HALT CHAIN between programs | MXU, VPU, sequencer |
| VLA elementwise | GELU / SiLU / sin / cos / mish: TFN; LayerNorm / RMSNorm: RSUM / RSSQ + RSQRT + MAC; residuals: ADD or the producing kernel's epilogue; RoPE: MUL + MAC with tables; time embedding: TFN sin/cos; Euler update x + dt * v: MAC | VPU (TFN, MAC: r5) |
| pixel shuffle, concat, Focus | stores into channel slices / strided DMA (as cnngraph does) | DMA |
| image preprocessing (VLM, VLA, YOLO) | normalization: MAC with per-channel constants; patches: strided loads; resize: a fixed-weight convolution / MAC; the host sends the decoded frame | VPU, DMA |
| vLLM sampling | use the engine's device sampler (`generate_card`) under vLLM's scheduler; refuse sampling features the device sampler lacks | VPU (ARGMAX, sampler) |
| BitNet | the engine path everywhere; the hybrid numpy model kept only as a reference | — |
| YOLO NMS | K iterations of: ARGMAX over scores, IoU of every box against the chosen one (MAX / MIN / MUL / RECIP), suppression as a 0/1 mask built arithmetically, scores *= mask | VPU |
| diffusion / flow loops (SD, VLA's Euler, Diffusion Policy) | the step loop in the program chain, the scheduler's coefficients as tables in DRAM | sequencer, VPU |
| several models at once | all images resident; a host scheduler (queue, priorities) starting programs at program boundaries; fixes vLLM's DeviceBusy | — |

### 2b. RTL candidates for r6

Ranked by what they remove or enable. None is required to get the host out of the
computation; they make the device versions efficient or exact.

| # | RTL change | Why | Unit | Size (estimate) |
|---|---|---|---|---|
| 1 | **TMM GSC**: an fp32 scale per weight row *and K block* (group of 128), applied to each block's integer sum before accumulation | exact 1-bit / ternary checkpoints with group scales (PrismML Bonsai: 1 fp16 scale per 128 weights); today a per-block TMM command each, without PAIR / QUAD | MXU epilogue | DSP per column; needs a resource study, the m8 build already uses 1299 of 1920 DSP |
| 2 | **TMM BIN**: 1-bit weights, 8 a byte (0 -> -1, 1 -> +1) | decode is DDR3-bound: an 8 B model is ~1 GB a token at 1 bit vs ~2 GB at our 2-bit packing; up to ~2x tokens/s (estimate) | MXU weight decode | small |
| 3 | **VOP CMP / SEL** (compare to a 0/1 mask; select A or B by a mask) | masks for NMS, top-p, thresholds, causal / padding masks without arithmetic tricks (the arithmetic step function is exact only for differences above ~1e-30) | VPU | small |
| 4 | **u8 input load** (LD of bytes to fp32, or QACT from u8) | camera frames cross PCIe at 1 byte a channel instead of 4 | DMA / VPU | small; check first whether the int8 row dequantization (`embed_host`) already covers it |
| 5 | **completion interrupt** (XDMA user IRQ / MSI) | the host waits without polling; lower CPU use and jitter for robot control loops | shell | small |
| 6 | **card-initiated reads of host memory** | MoE experts and models larger than 4 GB streamed by the card itself, no host CPU in the loop (today host-move through XDMA H2C) | PCIe / DMA | large; the author's `litepcie` branch (2026-10-15, not on main) has its own PCIe bridge and driver: evaluate it first |

Not proposed: a hardware sorter (ARGMAX in a loop covers NMS and top-k at these sizes), a larger
IMEM (HALT CHAIN chains programs), a JPEG decoder (decoding is I/O; cameras give raw frames).

## 3. Constraints that decide r6's shape

- r5's result decides the base: if m8 @ 150 MHz closes (`build_r5_timing.ps1`), r6 is m8 and
  every addition must fit the remaining DSP / LUT (synthesis of r5-m8 before the fixes: 290,999
  LUT of 298,600 sites, before packing; post-placement numbers pending). Otherwise r6 is m4.
- Each RTL change needs, before a build: Verilator RTL = ISA tests, the board model, and an
  out-of-context Vivado synthesis of the changed module checked for latches and RAM mapping
  (r5's TFN latches and T4 flip-flop buffer passed simulation and failed synthesis).
- DDR3 is 4 GB: resident images of SmolVLA (~0.5 GB int8), a YOLO (tens of MB) and a 1-bit 8 B
  LLM (~1.2 GB) fit together; a 27 B 1-bit model (~3.9 GB) does not.

## 4. Order

1. VLA graph on the device on r5 (2a, first three rows) with `--strict-fpga` for it.
2. Image preprocessing on the device; vLLM on the device sampler; NMS on the device.
3. The multi-model scheduler.
4. r6 RTL 3, 2, 1 (CMP/SEL, BIN, GSC) with their tests and OOC synthesis; then 4 and 5.
5. Evaluate `litepcie` for 6.
6. `--strict-fpga` everywhere: a run that computes anything on the host fails.
