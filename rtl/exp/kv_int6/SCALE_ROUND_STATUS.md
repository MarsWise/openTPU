# Experimental QST6 multiply and round stage

**MANUAL_REVIEW_REQUIRED.** This isolated stage multiplies a supplied FP32
value by a supplied FP32 inverse scale using the existing `otpu_fmul`, then
uses `otpu_fp::q8` and clamps to [-31,31]. Nonfinite operands or products
produce `invalid=1`, value zero. It does not compute row maxima or inverse
scales and is not connected to QST, MXU, runtime, or any bitstream.

All signals are synchronous to `clk`. A global enable freezes the multiplier,
valid pipeline, nonfinite flags and result during backpressure. Reset clears
local valid state and discards in-flight results; it does not undo any external
operation. Output-ready affects input-ready combinationally; integration must
prove there is no ready loop. X/Z behavior is unspecified. This fixed three-stage
wrapper has a peak rate of one value per cycle without stalls; it has not been
validated in the actual QST grant or memory protocol.

## Evidence

* `tests/test_kv_int6_codec.py::test_qst6_scale_round_pipeline_icarus` passed
  1,222 directed and seeded random FP32 pairs, including both signs of ties
  and neighbouring FP32 values. Expected multiplication uses
  NumPy binary32 with input/output FTZ, and rounding uses a separate integer
  binary32 oracle. The test exercises continuous traffic with source bubbles,
  backpressure, output stability, nonfinite operands, overflow and local reset.
  This sample is not exhaustive over all FP32 inputs and is not a complete
  QST6 row or model test.
* Vivado 2024.2.2 out-of-context synthesis of this source, `otpu_fp.sv` and
  `otpu_fpipe.sv` succeeded for `xc7k480tffg1156-2`: 206 LUT, 76 FF, 2 DSP,
  0 BRAM. Reports `build/logs/qst6_scale_round_ooc_{utilization,timing,drc}.rpt`.
  It has not been placed/routed. The DRC includes CFGBVS-1 and DSP pipeline
  warnings DPIP-1, DPOP-1 and DPOP-2. The OOC clock lacks `HD.CLK_SRC` and
  does not provide full-design timing. Per-lane duplication would cost extra DSP;
  evaluate reuse of QST's existing `u_q` multiplier before integration.
* Source SHA-256: `136aecf4e035f97ef7535c0df9daefeec6e3aaec1a4aafd592b9b041e261955e`.
* `bin/rtl-lint` / slang is unavailable; there is no independent signoff,
  formal proof, full-design synthesis, physical validation or model accuracy
  result. The findings below are source reasoning, **UNVERIFIED** unless covered
  by the specific tests/reports above.

## Review checklist and remaining risks

1. Functional intent: sampled comparisons pass; row-level integration absent.
2. Syntax: Icarus and Vivado accept; mandatory lint missing.
3. Synthesis: OOC only; actual hardware mapping in full QST unknown.
4. Sequential: one enable gates state and multiplier; reset discards data.
5. Combinational: output-ready propagates to input-ready and clock enable.
6. Latches: no procedural combinational block; mandatory latch lint missing.
7. State: three valid bits, no exhaustive reachability proof.
8. Case: no case statement.
9. Reset: local synchronous flush only; external drain is not implemented.
10. CDC: single-clock assumption, no CDC analysis.
11. Width: sampled output clamp; mandatory width lint missing.
12. Signedness: explicit signed q8 comparison; signedness lint missing.
13. Arithmetic: sampled FP32 product/round; scale generation is absent.
14. X/Z: outside contract; integration needs assertions and analysis.
15. Parameters: fixed multiplier latency of two cycles.
16. Protocol: sampled stalls/throughput and reset; formal liveness unproved.
17. Lint: mandatory hook unavailable.
18. DFT: not assessed.
19. Power: not measured; pipeline toggles with enable on empty cycles.
20. Maintenance: proposed format and nonfinite rejection are not ISA contracts.

Remaining risks: latency or valid alignment in full QST, ready loops, reset with
outstanding writes, handling of full-row metadata and errors, numeric accuracy,
unsupported X/Z, sign/width issues without lint, unconstrained clock and DSP
timing, integration hazards with DMA, missing model tests and independent review.
