# Model compatibility and end-to-end evidence

## r5x lean status per model (2026-11-23)

The current firmware is r5x lean 46a18fc1 (CFG_HASH add73610: no TMM BIN / BIN GSC / T5, no TFN
atan). "PHYS lean" rows ran on it on 2026-11-23 (r8_card_tests 17/0/0, r8_llm, r8_needle,
r8_speed, r8_ttft; ECC 0 / 0 except one r8_ttft step, VERIFIED_RESULTS.csv).

| Model | Level on lean | Decode (tok/s) | Left |
|---|---|---|---|
| Every r8 instruction but BIN / BIN GSC / T5 / atan, KV int8 / kv6 / k6v6 (Qwen3 tiny) | PHYS lean, == ISA | - | KV6 on a real model |
| Qwen3-0.6B | PHYS lean | 25.6 int8, 37.5 fp4 | 4-bit default after a quality gate |
| Qwen3.5-0.8B / 2B | PHYS lean | 20.1 / 8.4 int8, 28.0 fp4 (0.8B) | 0.8B fp4: one SEC step |
| LFM2.5-230M | PHYS lean | 66.9 int8, 96.7 fp4 | - |
| BitNet b1.58 2B4T, vLLM 4 slots (sampled / greedy / logprobs) | PHYS lean | 17.7 | 4-bit head |
| Bonsai 2 4B (GRQ QUAD) | PHYS lean | 8.5 (plain GRQ 3.4) | MXU 52 % MAC at 67 % DRAM: K-block accumulation in the MXU |
| Cactus Needle 26M int8 / CQ4 | PHYS lean, == ISA | 19.7 / 16.7 | decoder step at run arguments (one program a prompt): ISA / RTL, card NOT RUN; device token loop |
| LFM3 text decoder | PHYS lean | 8.3 | SWA / context past 255 |
| MiniCPM5-2B Q4_K_M | PHYS lean | 6.7 | chat template |
| Granite 4.2 3B Q4_K_M | PHYS lean | 6.2 | int8 MLP for accuracy |
| YOLOX-Nano, strict FPGA-only | PHYS lean (frame) | - | COCO-500 0.283 measured on r5x |
| Diffusion Policy pusht 10 / 100 steps, SmolVLA tiny / base | PHYS lean | - | device loop (12948 launches), program cache (prepare 39.6 s) |
| PaddleOCR-VL 1.6 | PHYS r5x (text right, ECC) | - | rerun with calibrate_until; preprocessing on the card |
| Bonsai 2 2B | ISA (T5 off on lean: its layers go TERNARY_GSC) | - | card run |
| Bonsai 1-bit (MLX) | NOT SUPPORTED on lean (needs TMM BIN) | - | - |
| Ouro 1.4B, Dory 4B, MiniCPM-V | ISA | - | card runs |
| Qwen3.5-4B, LFM2-2.6B, LFM2.5-8B-A1B, gemma-4, SmolLM3, Phi-4-mini | SOURCE INTEGRATED | - | first card run |
| ReasonLite 0.6B, FunctionGemma 270M | IMPLEMENTING | - | whole graph |
| MusicGen-small, Kokoro 82M, SD 1.5 | PLANNED | - | Kokoro: atan as a VPU polynomial (TATAN off) |

## r5x status per model (2026-11-21)

The current firmware is r5x c42beb65 (PROGRESS.md). "PHYS r5x" rows ran on it with ECC 0 / 0 in
that run (2026-11-19, VERIFIED_RESULTS.csv); acceptance still needs the long soak.

| Model | Level on r5x | Change against the r8 table below |
|---|---|---|
| Every r8 instruction (no BIN / BIN GSC), KV int8 / kv6 / k6v6 | PHYS r5x, == ISA | BIN / BIN GSC refused (not built) |
| BitNet 2B4T, vLLM 4 slots | PHYS r5x: 16.9 tok/s sampled, 17.0 greedy, logprobs max 8e-7 | ECC clean |
| Bonsai 2 4B | PHYS r5x: GRQ / GRQ QUAD logits bit-identical, 8.5 tok/s (GRQ QUAD) / 3.4 (GRQ) | - |
| Bonsai 1-bit (MLX) | NOT SUPPORTED on r5x (needs TMM BIN) | - |
| Qwen3-0.6B / Qwen3.5-0.8B / 2B / LFM2.5 | PHYS r5x: 25.6 / 20.1 / 8.4 / 66.9 tok/s; fp4 layers 37.5 / 28.0 / - / 96.7 | Qwen3.5-2B: 1 run of 3 had ch1 SEC |
| YOLOX-Nano | PHYS r5x strict FPGA-only: 4 detections | - |
| Diffusion pusht / SmolVLA base | PHYS r5x: rel L2 0.0015 (100 steps) / 0.0308 | SmolVLA wall fix NOT RUN on the card |
| Needle, LFM3, MiniCPM5, Granite | PHYS r5x: == ISA / PASS | - |
| PaddleOCR-VL 1.6 | ISA; card tool `tools/paddleocr_card.py` (NOT RUN) | - |

## r8 status per model (2026-11-16)

Levels as in the ТЗ (part 21). "PHYS exec" is execution on the card, not acceptance (ECC 0 / 0
over a long run is open; r8 2c800a8f has the MXU k0 erratum, see PROGRESS.md).

| Model | Checkpoint | Level | Host work left | Next |
|---|---|---|---|---|
| Qwen3 tiny, KV int8 / kv6 / k6v6 | synthetic | PHYS exec r8, == ISA | launches | KV6 quality on a real model |
| BitNet b1.58 2B4T (vLLM, 4 slots) | HF microsoft | PHYS exec r8 | tokenizer, scheduler | long run, ECC 0 / 0 |
| Bonsai 2 4B (GRQ) | prism-ml 2-bit | PHYS exec r8: logits; chat 3.43 tok/s (16.11; r5 0.68) | tokenizer | C2 decision; soak |
| Qwen3-0.6B, Qwen3.5-0.8B / 2B, LFM2.5-230M | HF | PHYS exec r8 (16.11): 25.6 / 20.1 / 8.4 / 66.9 tok/s decode; 0.8B had 1 SEC (not trusted) | tokenizer | repeat 0.8B; soak |
| Qwen3.5-4B, LFM2-2.6B, LFM2.5-8B-A1B, gemma-4, SmolLM3, Phi-4-mini | HF cache on the host | SOURCE INTEGRATED | - | first card run |
| YOLOX-Nano | Megvii | PHYS exec r8 (one frame), strict FPGA-only | image decode | COCO mAP on r8 (r5: 0.2825) |
| Diffusion Policy pusht | lerobot | PHYS exec r8 | 10790 launches / 100 steps | long run; device loop |
| SmolVLA base | lerobot | PHYS exec r8 | 12948 launches | action chunk buffer; device loop |
| Cactus Needle 26M (int8, CQ4) | Cactus-Compute | PHYS exec r8 (16.11): card == ISA, token picked on the card, ECC 0 / 0 | tokenizer, a launch a token | device token loop |
| LFM3 text decoder (24 layers) | lfm3-omni, rows extracted | PHYS exec r8 (16.11): cos 0.9999, 8.26 tok/s picked on the card, ECC 0 / 0 | tokenizer | chat, long context (past 255), soak |
| MiniCPM5-2B | GGUF Q4_K_M | PHYS exec r8 (16.11): cos >= 0.9992, 6.73 tok/s, ECC 0 / 0 | tokenizer | chat template, soak |
| Granite 4.2 3B | GGUF Q4_K_M | PHYS exec r8 (16.11), MLP NVFP4 / FP4: cos 0.89 (BOS) .. 0.99, 6.18 tok/s, ECC 0 / 0 | tokenizer | int8 MLP on the card (accuracy) |
| Bonsai 2 2B | PTQ1_0 (T5) | ISA (real weights) | tokenizer | card |
| Ouro 1.4B | HF | ISA (8 layers int8; 24 with int4 MLP) | - | 24 layers int8, early exit |
| Dory 4B | GGUF Q5_K_M | ISA (subnet 4 + 16 + 4) | - | whole model |
| PaddleOCR-VL 1.6 | HF | ISA (reads a rendered line) | preprocessing, row feed | on-device preprocessing |
| MiniCPM-V | HF | ISA (tests) | - | answer on a real image |
| ReasonLite 0.6B, FunctionGemma 270M | HF | IMPLEMENTING | - | whole graph |
| MusicGen-small, Kokoro 82M, SD 1.5 | HF | PLANNED (blocks only) | - | - |

## New workload families requested

Updated 2026-10-19 (feature HEAD after the `9bd4d83c` upstream merge).

| Family | Selected checkpoint / revision | Current evidence in this branch | Status |
|---|---|---|---|
| BitNet b1.58 | `microsoft/bitnet-b1.58-2B-4T` (sha256 `8143ae11…3975b`, offline `AutoBitLinear`) | **On the card (r3, 2026-10-20):** the whole 30-layer model, prompt "The capital of France is": greedy 16/16 tokens equal to HF transformers (hybrid and the one-run-a-token engine), first-token cosine 0.9958 / 0.9923 vs HF; 1 layer card = ISA bit for bit; engine 145.8 ms card / token (271.5 ms wall, PCIe x4), TMM 124.9 ms. BITNET_COMPAT.md | `PHYSICAL FPGA VERIFIED` (greedy generation, logits vs HF) |
| YOLO / CNN | **YOLOX-Nano** (Megvii-BaseDetection/YOLOX, Apache-2.0; `yolox_nano.pth` sha256 `cd28f55fbbc1829f99d9ac9b38a16d259a22889739c8728ea877610201feff7b`; 416², 0.91 M params, COCO mAP 25.8 per its README) | `opentpu/kernels/conv.py` (Conv2D implicit GEMM on the MXU; depthwise, max pool, upsample on the VPU), `tools/yolox_run.py` (the model's own PyTorch forward with every Conv2d / MaxPool2d / Upsample on the kernels). ISA simulator, 416² `assets/dog.jpg`: 113 convolutions = 100% of the 522.6 M MACs on the device; detections dog 0.843 / car 0.817 / bicycle 0.812 vs float 0.829 / 0.809 / 0.808 (the float model's 4th, cat 0.380, is missed); per-layer error 1.2-2.4% on float inputs, 19% median accumulated over the graph (uncalibrated int8). Whole graph on the device (`opentpu/cnngraph.py`, `tools/yolox_graph.py --strict-fpga`): 126 device ops incl. box decode; host = input write, readout, NMS. With `--wsplit` (int8 hi+lo weights): head rel L2 0.092, all 4 float detections (IoU ≥ 0.977). **COCO val2017, first 500 images (`tools/yolox_coco.py`, host run 2, 2026-10-20, card r3 0xe9727303): mAP50-95 float 0.2844, emul 0.2840, emul --wsplit 0.2828, card 0.2821 (0.44 s/image), card --wsplit 0.2852 (0.41 s/image)** — the card at the float model's accuracy | `ISA SIMULATED (whole model)`; kernels `RTL SIMULATED`; **COCO mAP `PHYSICAL FPGA VERIFIED`** (per-launch card = ISA check of the whole graph: re-run in claude_run.sh v10) |
| Vision-language | MiniCPM-V (upstream `opentpu/llm/minicpmv.py`, `opentpu/kernels/vit.py`) | Upstream code and tests in the branch. `tests/test_minicpmv.py` on the merged tree (2026-10-20, ISA simulator, no Verilator): 39 passed, 1 failed — `test_warm_entries_are_found_by_the_first_start` fails identically on pristine upstream `9bd4d83c` (program-cache disk hits 26 vs 13: an upstream issue, not NextGen). Image-conditioned check on a real checkpoint not run here | `UPSTREAM CODE; SOFTWARE TESTS RUN (1 upstream failure)`; real model `NOT RUN` |
| Vision-language-action | **SmolVLA** (LeRobot `7abbc995`, Apache-2.0; checkpoint `lerobot/smolvla_base` to run on the host) | `opentpu/torchdev.py` + `kernels/linear.py`: the policy's own `sample_actions` (SigLIP tower, connector, VLM prefix + KV cache, the action expert's 10 Euler steps) with every nn.Linear and the patch-embedding conv as device kernels, the rest counted on the host. `tools/smolvla_run.py --tiny` (random tiny SmolVLA built offline at the real architecture; ISA simulator): 100% of MACs on the device (linears, patch conv, every attention via `kernels/sdpa.py`; host: elementwise / norm ops only), actions rel L2 0.021 vs float32 (stage 1 with host attention: 96.1%, 0.016); `tests/test_torchdev.py`. **Card (host run 2, r3): the tiny policy, 553 kernel runs, every one equal to the ISA simulator bit for bit, 100% of MACs on the device, actions rel L2 0.0205 vs float32, 3.8 M card cycles.** `smolvla_base` on the card failed to compile (a 12,288-wide linear: ACT_ROWS rows did not fit TMEM; fixed, the kernel takes fewer rows; re-run in v10). Inputs synthetic (recorded LeRobot frames: next) | tiny **`PHYSICAL FPGA VERIFIED`** (card = ISA, framework compared); `smolvla_base` `NOT RUN` |
| vLLM | **vLLM 0.32.0** out-of-tree platform `opentpu/vllm` (entry point `vllm.platform_plugins`) | Qwen3 / Llama-likes with KV slots, BitNet b1.58 with one slot (its `BitNetForCausalLM` architecture and `bitnet` quantization registered as stand-ins vLLM only inspects); vLLM's scheduler, continuous batching, preemption (recompute), abort, streaming OpenAI server; greedy tokens and sample logprobs == openTPU Engine; unsupported sampling features refused (400). LFM2 / Qwen3.5 / Gemma 4 / MoE / VLM refused. `tests/test_vllm_plugin.py` (9 passed). Host run 2: the tests ended in vLLM's shutdown on the host's CPU-only torch (`torch.accelerator.empty_cache`) and the card smoke stopped at a wrong XDMA node check; both fixed | `SOFTWARE VERIFIED` (ISA simulator); card `NOT RUN` (v10) |
| MoE / MTP | Upstream `opentpu/llm/moe.py`, `mtp.py`, offload (`docs/offload.md`) | Upstream code and tests in the branch. Re-run on the merged tree (2026-10-20, ISA, no Verilator): `test_mtp` 38 passed, 4 skipped (Verilator); the MoE suites: see PROGRESS.md | `UPSTREAM CODE; SOFTWARE TESTS RUN` (partial) |

## Existing LLM support

The repository README and model-specific docs record existing on-card LLM operation, including
INT8 and 4-bit formats for named model families. Those are upstream baseline claims and were not
retested in this task. This branch makes no new claim about model accuracy, checkpoint
compatibility, RTL execution fraction, or device performance.

Before adding a row as supported, record the exact checkpoint repository/revision and file hashes,
converter/quantizer, tokenizer/processor, operator graph, hardware execution locations, golden
reference comparisons, and reproducible run command.

## Diffusion Policy (LeRobot) -- 2026-10-26

`tools/diffusion_run.py`: LeRobot's Diffusion Policy (ResNet-18 image encoder with GroupNorm and
spatial softmax, conditional 1-D U-Net, DDPM) with every nn.Conv2d, nn.Conv1d,
nn.ConvTranspose1d and nn.Linear on the device (torchdev `DeviceConv1d` / `DeviceConvTranspose1d`:
im2col / col2im on the host, the GEMM on the MXU). GroupNorm, Mish, spatial softmax and the
scheduler's update on the host (counted). ISA SIMULATED: a random small policy (12.3 M params,
5 steps): 100.00% of the MACs on the device, actions rel L2 0.018 vs float32
(`tests/test_conv1d_diffusion.py`). lerobot/diffusion_pusht on the card: `models_r5.sh` (not run
yet). Image diffusion (a U-Net / DiT for pictures) is not in the tree.
