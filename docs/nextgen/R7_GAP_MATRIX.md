# r7 gap matrix: merged baseline (2026-10-30)

The r7 planner branch `0446b552` was merged with r6 `20a49288` at
`9b399c51`; their common hardware ancestor is `45b1c1d3`. This table records
what the source and prior evidence support, not a promise of FPGA-only model
execution. r5 `45b1c1d3` was physically tested; there is no routed or physical
r6/r7 result. `55b975c1` is a report commit, not a different bitstream.

| Area | Current implementation / evidence | Missing acceptance evidence or work |
|---|---|---|
| DDR3 / ECC | r5 bounded bring-up, memory patterns and model runs had 0 corrected and 0 uncorrectable ECC on both channels; earlier separate runs did see channel-0 errors | Identify cause under failing workloads; test long runs, partial writes and ECC without hiding internal counter clears; do not generalize bounded r5 evidence to r7 |
| MXU and r6 instructions | r5 TMM/PAIR/QUAD and ternary GSC; r6 adds CMP/SEL, BIN, BIN+GSC and VU8. ISA/RTL tests and block-level synthesis exist | Full r7 synthesis, placement, routing, timing and physical tests; BIN PAIR/QUAD absent; profile GSC bandwidth before modifying the MXU |
| Runtime / memory | `opentpu/nextgen/planner.py` implements a lifetime allocator and declared placement trace. `cnngraph.build(..., reuse_memory=True)` can now use it for the real CNN buffer arena; small graphs and residuals have ISA tests | The default remains the original layout. Opaque kernels and buffers with borders/tails/fill are pinned; weights and program staging are laid out separately. Validate real YOLOX and the board before broader reuse; strict trace alone does not detect undeclared CPU math |
| Resident execution | ISA and board support HALT CHAIN, and existing LLM paths use device programs | General resident graph/loops and dependency handling for actual model graphs, measured host/PCIe/DDR overhead |
| Image input / YOLOX | r6 has uint8 conversion, affine normalization, patch layout and device NMS; separate ISA/RTL tests; r5 YOLOX graph vs ISA ran on one image | Device resize/letterbox and model integration; r6 NMS and the complete image path have not run on physical FPGA; compare boxes, accuracy and FPS |
| MiniCPM-V | Vision tower/projector and language-side code exist; patchification primitive has ISA/RTL coverage | Raw image integration, resize, patch/position handling, complete real image-conditioned output, memory feasibility and physical validation |
| SmolVLA | Split / `torchdev` paths and kernels exist; r6 fixes one wide elementwise TMEM failure | Real `lerobot/smolvla_base` full resident graph, expert and action loop, reference comparison; older real base run failed and is not a pass |
| Diffusion Policy PushT | r5 real 10-step case ran with CPU numerical work and clean boundary ECC, rel L2 0.00191353 vs saved CPU actions | Device GroupNorm, spatial softmax, scheduler, denoising loop and resident graph; run real policy and compare actions and device vs wall time |
| LLM / BitNet / Bonsai | r5 BitNet 2B and Bonsai 2 4B passed bounded CPU-reference checks on physical FPGA | Regress new r7, measure bandwidth and long runs, optimize GSC only against profiling; do not extend first-token checks to full generation |
| vLLM | r5 one-request greedy test matched Engine; r6 strict path does greedy device picks | Device sampling/RNG/logprobs, multi-slot and lifecycle validation; verify strict path against hidden host sampling and unsupported features |
| SD 1.5 / Kokoro / MusicGen | No complete real-model FPGA validation in this branch | Measure checkpoint, buffers, peak memory and traffic; implement graph and run end-to-end before reporting support |
| Build / timing | `tools/build_r6_133.ps1` and Vivado 2024.2.2 configuration exist | Dedicated r7 reproducible build, full route and DRC/CDC/methodology/I/O reports; physical programming requires separate user permission |

The latest bounded r5 bitstream had routed WNS +0.050 ns and WHS +0.016 ns
under its specified constraints. It also had CDC/methodology and external I/O
constraint findings that require review; these numbers are not r7 timing.
The board has 4 GiB DDR3, and planned memory must include activations, KV,
weights, programs, scratch, DMA staging and alignment, not only checkpoint size.

CPU tokenization, orchestration and file/network I/O are permitted. CPU
numerical inference must be identified and rejected for an FPGA-only claim.
