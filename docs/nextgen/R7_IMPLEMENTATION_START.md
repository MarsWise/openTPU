# r7 implementation, first code increment

Base: verified r5 commit `55b975c14346cc692f1765729d750d79e8bc0b04`.
Branch: `feature/r7-fpga`. Existing r5 and r6 trees remain unchanged.

## Implemented here

- `opentpu.nextgen.plan_buffers`: first-fit, lifetime-aware DDR3 allocation;
  explicit aligned resident weights and reserve, capacity checks, collision
  verification. Intervals are half-open.
- `opentpu.nextgen.ExecutionTrace`: explicit placement tracking and
  rejection of declared CPU numerical operations in strict mode.
- Pure Python regression tests in `tests/test_nextgen_planner.py`.

Neither feature is connected to `torchdev.py`, `cnngraph.py`, vLLM, or the
board executor yet. This commit does not add new FPGA operators or models,
and does NOT establish hardware-only execution by itself.

## Next implementation dependencies

1. Integrate r6 CMP/SEL, BIN+GSC, VU8 and NMS via reviewed source changes.
2. Feed compiler-produced real tensor lifetimes into the DDR allocator and
   reserve IO/KV/scratch/workspace explicitly. Add graph trace instrumentation.
3. Implement FPGA kernels for missing ops (GroupNorm, Mish, im2col/col2im,
   image resize, ConvTranspose1D), each matched to RTL/ISA/torch references.
4. Connect resident graph execution to real Diffusion PushT and SmolVLA.
5. Add verified model adapters for SD1.5 (CLIP/U-Net/CFG/scheduler/VAE),
   Kokoro, and MusicGen, with real checkpoint format/size validation.
6. Physical validation: routed timing at 133.33 MHz, resource utilization,
   boundary and internal ECC checks, tensor accuracy and wall/device timing.

Keep stable DDR3 unchanged. Do not program FPGA or modify upstream.
