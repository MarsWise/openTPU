# Architecture audit at the pinned upstream base

## Scope and evidence

- Base: `FeSens/openTPU/main`, `5309a20d28948fa910897e85a96f82db6ee5c1fd`, fetched
  2026-10-14.
- This is a source inspection, not a new performance run. Existing benchmark values below are
  quoted only as documented project results.
- Board values in `targets/ypcb-00338.toml` and `docs/board.md` describe the project target;
  board identity and electrical details still require confirmation by the user.

## Existing execution path

```text
Host Python / model kernels
  -> compiler (`opentpu/compiler.py`, `opentpu/isa.py`)
  -> program + operands in DRAM
  -> sequencer (`rtl/seq/otpu_seq.sv`)
       -> DMA / DRAM (`rtl/dma/`, board LiteDRAM integration)
       -> quantizer (`rtl/vpu/otpu_quant.sv`)
       -> matrix unit (`rtl/mxu/otpu_mxu.sv`)
       -> vector/stream units (`rtl/vpu/`, `rtl/seq/`)
       -> TMEM (`rtl/mem/otpu_tmem.sv`)
  -> simulator (`opentpu/isasim.py`) and Verilator runner (`opentpu/rtlsim.py`)
```

`rtl/top/otpu_pkg.sv` defines opcodes, unit assignment, memory ranges and the scoreboard contract.
`OP_MM` is served by the existing MXU; `OP_QACT`/`OP_QST` use the quantizer; `OP_VOP` handles
vector operations; DMA and stream operations are separately scheduled. The compiler, ISA
simulator, RTL and runtime already share quantization/layout rules documented in `docs/isa.md`,
`docs/quant.md`, `docs/mx_mxu.md` and `docs/stream.md`.

## Target facts and limits of this audit

The repository's YPCB-00338 target file declares XC7K480T, two DDR3-1066 channels, PCIe Gen2 x8,
133.33 MHz, `D=128`, `MCOLS=4`, and a 64K-word TMEM default variant. These are configured/project
facts, not independently measured facts from the physical card. This source audit has no
utilization or timing report to inspect; the separate read-only JTAG discovery is in
`HARDWARE_CONNECTIVITY.md`.

The README records historical on-card LLM results and documents the current image as a four-column
systolic design. Those benchmark descriptions are not recomputed here. This audit has not measured
rooflines, RTL critical paths, live DDR contention, area, post-route Fmax, or performance for any
new workload.

## Work still required

1. Trace current RTL hierarchy, clock/reset domains, DMA arbitration, each TMEM port, and board
   source manifests using the exact chosen profile.
2. Run RTL/ISA differential and Lens traces for selected representative checkpoints.
3. Obtain user-provided Vivado utilization and timing reports before establishing a resource
   margin or promoting a specialized core.
4. Select actual BitNet/Tern2, CNN, VLM, VLA and serving models before claiming operator coverage.
5. Keep physical measurements in a distinct evidence class from source inspection and simulation.

## Current feature-branch update (2026-10-16)

The source-inspection audit above is intentionally pinned to the earlier base and remains a
historical snapshot. The current published feature branch is
`668dcff5295043a2618e558fe72c44a183ebd56d`, integrating upstream `f237dab2` through merge
`07eb6cd1`.

- Native generic ternary TMM (`OP_TMM=0x23`) is now wired through ISA encoding/simulation,
  compiler lowering, MXU sequencer/footprints and packed-weight PE execution. This is not
  checkpoint-specific BitNet/Tern2 support.
- The same MXU continues to carry the established INT8, INT4, FP4 and NVFP4 modes. TMM uses
  signed add/subtract/skip selection while existing multiplier logic remains; no DSP saving is
  claimed.
- Current targeted verification is seven D=128 TMM RTL-vs-ISA tests and 79 compiler/ISA/ternary
  tests (see `PROGRESS.md`). The complete post-upstream test suite and model-framework numerical
  validation remain outstanding.
- Read-only JTAG discovery confirms the local device part/IDCODE, but does not establish the
  loaded design, PCIe endpoint, driver, DMA or inference state. See `HARDWARE_CONNECTIVITY.md`.
- No resource/timing estimate or physical result exists for the TMM change. The user-owned target
  remains the repository-declared XC7K480T configuration; board declarations are not a fresh
  physical measurement.
