# Experimental QST6 group write

**MANUAL_REVIEW_REQUIRED.** `otpu_qst6_group_write` is an isolated composition
of the experimental FP32 multiply/round stage, INT6 formatter and slice-idle
fence. It captures 16 FP32 lanes and a supplied inverse scale, processes them
sequentially, then emits three full 32-bit words. A nonfinite input or product,
an unaligned address or an address above `0xfffffff4` rejects the whole group
before any word is issued. An accepted word is not a memory completion; done
depends on the external slice-level idle contract.

This is **not** an integrated QST, KV cache, ISA instruction or bitstream. It
does not calculate amax/scale, store scales, validate the actual allocated DRAM
region, provide memory error responses, enforce scoreboard hazards, or handle V
transpose. Reset clears only local state: quiesce/drain external writes before
reset or invalidate the affected data. All signals use `clk` and X/Z behavior
is unspecified. Only one group can be outstanding; the 512-bit input capture
and sequential multiplier are a correctness prototype, not a throughput design.

## Evidence

* `tests/test_qst6_group.py` and Icarus test 33 groups, 528 FP32 lanes, four
  supplied inverse scales, tie rounding, output stalls and stability, changing
  inputs after acceptance, simulated global idle, end-of-4-GiB boundary, all 16
  nonfinite lane positions, nonfinite inverse, product overflow, address
  rejection, reset during calculation and a subsequent group. Expected values
  use NumPy and an independently constructed six-bit layout. The memory fence
  is driven by the testbench, not the real `otpu_native_dram` or `otpu_slice`.
* Vivado 2024.2.2 out-of-context synthesis, `xc7k480tffg1156-2`, completed:
  588 LUT, 898 FF, 2 DSP and 0 BRAM. Reports are
  `build/logs/qst6_group_write_ooc_{utilization,timing,drc}.rpt`.
  This is not placed/routed. DRC includes CFGBVS-1 and DSP pipeline warnings
  DPIP-1, DPOP-1, DPOP-2; the OOC clock lacks HD.CLK_SRC. The area includes a
  captured 512-bit input; it is not a full-r8 resource or timing estimate.
* Source SHA-256:
  `03df7299f989e5fce09d6605a1eb6547270160ce461c908912c1aa1096d8a02d`.
* `bin/rtl-lint` / slang is unavailable. No independent RTL signoff, formal
  proof, integrated memory simulation, route or physical/model test exists.

## Review categories

The following are source-level reasoning, **UNVERIFIED** beyond the named tests:

1. Function: selected cases pass; scales and real QST are absent.
2. Syntax: Icarus/Vivado accept; required lint missing.
3. Synthesis: only isolated OOC, real QST mapping unknown.
4. Sequential: state and payload use nonblocking assignments.
5. Combinational: grant/ready path must be checked at integration.
6. Latches: no procedural combinational block; lint missing.
7. FSM: default returns to IDLE; illegal-state recovery is not formally proved.
8. Case: explicit default; fault behavior with a busy writer is not proved.
9. Reset: external writes cannot be cancelled, quiescence is required.
10. CDC: same clock assumed; no CDC analysis.
11. Width: counters hold 0..16; address wrap guard tested, lint still missing.
12. Signedness: delegated to scale/codec; signedness lint missing.
13. Arithmetic: sampled NumPy reference; special-value semantics incomplete.
14. X/Z: outside contract; assertions required at integration.
15. Parameters: fixed group size 16 and fixed writer protocol.
16. Protocol: abstract stalls/fence tested, no liveness or ownership proof.
17. Lint: mandatory hook unavailable.
18. DFT: not assessed.
19. Power: not measured.
20. Maintainability: layout/limitations documented; production ISA unresolved.

Outstanding risks include ready loops, memory RMW conflicts, reset after a
partially accepted group, stale idle, missing error propagation, unknown/X
inputs, missing formal reachability and arithmetic proof, additional DSP/FF
cost, absence of row-scale alignment and actual QST scheduling, and absence of
full timing, model accuracy and physical evidence.
