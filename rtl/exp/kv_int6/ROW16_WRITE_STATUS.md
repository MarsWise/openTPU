# Experimental QST6 16-value scale-to-write controller

Update: the controller also has compile-time `AUTO_AMAX=1`, scanning its
captured 16 lanes; the default 0 keeps the caller-supplied amax interface.
See `ROW16_AUTO_STATUS.md` for the new test and synthesis evidence; the OOC
figures and source state below are historical pre-parameter results.

**MANUAL_REVIEW_REQUIRED.** `otpu_qst6_row16_write` accepts 16 FP32 values,
a supplied amax and a byte address, invokes the INT6 scale guard, then passes
the inverse scale and captured values to the experimental group writer. The
three data words use the slice-idle fence; the computed scale is presented with
`done`. Unsupported amax, a nonfinite lane/product, or an invalid alignment or
32-bit address rejects without a data write in the tested cases.

The caller must supply the **actual** maximum for the intended quantization
row: the controller does not verify or calculate amax. It does not write the
scale to DRAM, allocate memory, check the actual device's memory limit, manage
scoreboard hazards, report memory write failures or guarantee atomic scale/data
publication. The controller handles one group, not an arbitrary-length QST
row or transposed V cache. QST, ISA, MXU, firmware and actual memory interfaces
do not instantiate it. Reset clears local state but cannot cancel accepted
external writes; quiesce/drain and invalidate as required. All ports use one
clock; X/Z and asynchronous resets are unspecified.

## Evidence

* `tests/test_qst6_row16.py` uses Icarus to compare 16 random/zero groups and
  their scale with a Python reference. It changes inputs after acceptance,
  applies write stalls, checks stable outputs and a testbench-driven shared-idle
  fence, checks invalid maxima, nonfinite data and bad addresses, and resets during scale calculation
  before running a fresh group. Amax and memory-idle are supplied by the testbench.
  It does not instantiate the real `otpu_quant`, native DRAM, scoreboard or DDR.
* Source SHA-256:
  `bc8d1758704588c949e8cacc81a1ac3c3c7ed8987f5400c2377ca0452518ef53`.
* Vivado 2024.2.2 OOC synthesis for `xc7k480tffg1156-2` completed: 2953 LUT,
  3412 FF, 18 DSP, 0 BRAM. This includes both the entire Q6 scale core and
  duplicated FP32 capture registers; it is not the incremental area in r8.
  Reports are `build/logs/qst6_row16_ooc_{utilization,timing,drc}.rpt`.
  DRC includes CFGBVS-1 and DSP pipeline warnings DPIP-1, DPOP-1 and DPOP-2;
  clock source is unset and there is no placement/routing or timing closure.
* The mandatory `bin/rtl-lint` / slang is unavailable. No independent signoff,
  formal/CDC proof, production QST regression, full design implementation or
  physical/model test exists.

## Review categories (source reasoning is UNVERIFIED)

1. Function: sampled groups match; caller supplies true amax.
2. Syntax: Icarus accepts; mandatory lint unavailable.
3. Synthesis: see OOC reports, no full QST mapping.
4. Sequential: one synchronous state block with nonblocking assignments.
5. Combinational: ready/valid and memory paths need integration review.
6. Latches: no procedural combinational block; lint unavailable.
7. FSM: four states; no formal reachability or recovery proof.
8. Case: default returns idle, downstream busy-state fault recovery unproved.
9. Reset: stale scale masked in selected test; memory writes not cancellable.
10. CDC: all signals assumed synchronous; no CDC analysis.
11. Width: 512-bit captured input, tested addresses, mandatory width lint absent.
12. Signedness: delegated to QST6 scale/group modules, lint absent.
13. Arithmetic: dependent on caller-provided amax and existing FP32 semantics.
14. X/Z: unspecified; integration assertions are required.
15. Parameters: fixed 16-lane row and fixed scale pipeline.
16. Protocol: sampled backpressure/fence; no formal liveness proof.
17. Lint: required hook unavailable.
18. DFT: not assessed.
19. Power: not measured; data is stored in multiple intermediate registers.
20. Maintainability: experimental contracts and missing pieces are documented.

Key open risks: incorrect or stale amax, untracked partial writes, scale/data
visibility and error propagation, ownership under DMA/host accesses, stall or
reset interaction, lack of actual QST/DDR tests, high duplicated storage and
DSP cost, timing, and numerical quality for real model KV caches.
