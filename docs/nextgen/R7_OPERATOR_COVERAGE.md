# r7 operator coverage and evidence, 2026-10-30

This is an implementation audit, not a traced, exhaustive list of every node
in every checkpoint. A complete operator list needs the exact checkpoint and
input shapes plus captured graphs. `device primitive` means an implementation
exists; it does not mean it is connected to that model. r5 board evidence is
for commit `45b1c1d3`, not for the merged r7. The current r7 build is pending.
Details and limitations of physical r5 cases are in
`R5_45B1_BOARD_VALIDATION_2026-10-30.md`.

| Model / numerical operations | Current placement and evidence | Missing for FPGA-only / acceptance |
|---|---|---|
| YOLOX-Nano: resize, letterbox, input conversion, convolution, pooling, upsample, residual/concat, decode and NMS | r5 graph was checked FPGA vs ISA for one image. r6 adds device uint8 conversion / normalization, patch layout and NMS with separate ISA/RTL tests. | Device resize/letterbox and wiring the new input/NMS primitives into the actual model; independent accuracy and latency validation. |
| Diffusion Policy PushT: image encoder, conv1d, normalization, activation, spatial softmax, denoising and scheduler | r5 ran ten steps against saved CPU actions, rel L2 0.00191353. `tools/diffusion_run.py` performs im2col/col2im, GroupNorm, Mish, spatial softmax and scheduler on host. A standalone DDPM epsilon step now has ISA vs CPU/diffusers and RTL vs ISA tests. | Connect every operation and DDPM step to the real policy, support the checkpoint scheduler settings, RNG and device loop, then compare full actions. |
| SmolVLA base: image tower, connector, attention, norms, RoPE, activations, action expert and Euler loop | `torchdev` offloads supported linear, patch convolution and attention; the split runner is explicitly hybrid and refuses strict FPGA-only mode. A wide elementwise TMEM allocation bug was fixed. | Real base graph and action loop resident on device, including host ops, image handling and end-to-end board comparison. Older real base failure is not a pass. |
| MiniCPM-V: image resizing, normalization, patches, vision tower, projector, LLM | Resize and preprocessing in `opentpu/llm/minicpmv.py` include host numerical work; r6 input/patch device primitives have separate tests. | Connect device resizing and preprocessing, position/layout handling and all tower/projector/LLM operators to a real checkpoint; memory and board validation. |
| BitNet b1.58 2B: ternary linear, norm, attention, KV, logits and generation | Native r5 ternary path; a bounded eight-token board run checked first logits and first token against CPU. | Check all generated tokens and complete numerical execution, r7 regressions, sampling and sustained performance. |
| Bonsai 2 4B: Hadamard, ternary group-scaled linear, transformer, logits | r5 bounded full prompt comparison: logits cosine 0.999926239, next token matched. Hadamard uses VPU instructions. GSC has serial group issue; the synthetic profiling tool is ISA-tested only. | Normal chat picks the first token on the host, and sampled decoding consumes host-generated uniform values. Move these to the device, profile real weights and traffic, optimize with per-group scaling preserved, validate long generation and r7. |
| vLLM: attention/KV, sampling, logprobs, multiple slots | r5 single request matched an openTPU Engine. r6 strict path provides greedy device selection; general sampling/logprobs have host paths. | Device RNG and sampling beyond greedy, logprobs, multi-slot lifecycle and audit of actual numerical backends. |
| Stable Diffusion 1.5: text encoder, U-Net, CFG, scheduler, VAE | No complete model graph or end-to-end FPGA result established in this audit. | Trace exact checkpoint, implement and connect missing ops, budget memory/traffic, validate output against reference. |
| Kokoro-82M: text/phoneme input, model graph and waveform generation | No complete real-model FPGA result established in this audit. | Trace exact checkpoint and input boundary, implement graph and numerical waveform path, measure memory and output accuracy. |
| MusicGen Small: text encoder, autoregressive audio tokens, codec decode | No complete real-model FPGA result established in this audit. | Trace exact checkpoint and sampling/codec graphs, device implementation, memory budget and audio/reference tests. |

## Common reusable operations

| Operation | Existing evidence and boundary |
|---|---|
| FP32 arithmetic, reductions, LayerNorm/RMSNorm, TFN, device loops | ISA/compiler and various existing RTL tests; model-specific wiring remains to be audited. |
| CMP/SEL, binary and binary group-scaled TMM, uint8 input | r6 ISA/RTL or block synthesis evidence; no routed r7 or physical validation. |
| DDPM epsilon step with supplied noise and coefficients | `kernels/ddpm.py`: standalone compiler kernel, CPU/diffusers comparison and one RTL/ISA comparison. Not integrated into a real model, no RNG and no resident scheduler loop. |
| Random generation | `rtl/exp/philox` contains a standalone RTL-tested Philox4x32-10 block; not connected to the ISA, model execution or current bitstream. No Gaussian generator or per-request runtime state yet. |
| Memory reuse | `cnngraph.build(..., reuse_memory=True)` integrates the lifetime planner for CNN activations. Conservative pinning and separate weights; small ISA graphs tested, real model and hardware validation pending. |
| Strict execution | `--strict-fpga-only` fails closed in the hybrid SmolVLA, Diffusion, YOLOX and chat runners. YOLOX's older `--strict-fpga` covers the graph/decode only and does not establish strict preprocessing or NMS. The older chat flag misses first-token selection and host RNG. There is no demonstrated universal runtime audit capable of detecting every CPU numerical fallback. |

Acceptance evidence must distinguish implementation, ISA test, RTL test,
synthesis, routed timing and physical validation. A model is not FPGA-only
merely because its matrix multiplications use the FPGA.
