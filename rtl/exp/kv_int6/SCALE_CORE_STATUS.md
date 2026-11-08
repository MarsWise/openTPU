# Experimental INT6 QST scale arithmetic

**MANUAL_REVIEW_REQUIRED.** The existing `otpu_qscale` in
`rtl/vpu/otpu_quant.sv` now has a compile-time `Q6` parameter, default 0.
`Q6=1` selects FP32 constants 31 and 1/31 for inverse and scale. The existing
QST instantiation does not enable it; `Q6=0` retains selection by `q4`.
This is not a dynamic Q4/Q6 selector, an ISA feature, or a change that puts
INT6 in a bitstream. The existing package constants in module port expressions
were qualified explicitly so Icarus does not create implicit one-bit wires.

`otpu_qst6_scale_guard` is an isolated wrapper using `Q6=1`. It accepts +0 or
positive finite maxima in `[2^-121, 2^126)`; negative values, subnormals,
nonfinite values and values outside that range return invalid and zero data.
All ports share one clock. A transfer occurs when valid and `en` are high on a
clock edge. `en` stalls the entire pipeline. Synchronous reset clears the
wrapper's valid/bad delay, rejecting stale results; it does not cancel external
memory transactions. The wrapper uses a fixed depth 32 for LM=2/LA=4 and is
not parameterized. X/Z and asynchronous reset/clock inputs are unsupported.

## Evidence and limits

* `tests/test_qst6_scale.py`: Icarus compiles the exact qdly/qscale module text
  extracted from the shared source (with a compilation-unit package import,
  because Icarus cannot elaborate unrelated `otpu_pkg` functions). 1,258 input
  magnitudes per mode cover every safe FP32 exponent, representative fraction
  boundaries, randomized fractions, zero, tags and stalls. Experimental Q6
  and legacy Q4/INT8 outputs match the project's FP32 reference. This is sampled
  unit validation, not the full production quantizer regression.
* A separate guard test exercises valid/invalid values, exact range minimum,
  NaN, infinity, signed/denormal inputs, stalled clock enable and reset while
  disabled with an older transaction in flight. It checks old output is not
  accepted as a new result. This is not a formal reset or CDC proof.
* `tests/test_qst6_group.py` glues guard and group writer in a testbench and
  compares eight 16-value rows, including zero, with the Python row reference.
  The testbench supplies row maxima and controls transfer/fence; there is no
  autonomous row controller, QST/MXU hookup or actual memory controller.
* Vivado 2024.2.2 out-of-context synthesis on `xc7k480tffg1156-2`:
  Q6 `otpu_qscale` alone: 2492 LUT, 1877 FF, 16 DSP; guard plus core:
  2524 LUT, 1911 FF, 16 DSP; both 0 BRAM. Reports are
  `build/logs/qst6_scale_{core,guard}_ooc_{utilization,timing,drc}.rpt`.
  These are entire isolated cores, not incremental r8 cost. They are not placed
  or routed; HD.CLK_SRC is unset and DRC includes CFGBVS-1 and DSP pipeline
  warnings DPIP-1, DPOP-1 and DPOP-2.
* SHA-256 `rtl/vpu/otpu_quant.sv`:
  `7c66ed181e9ee5b04a56e9c616c32db240afc4512e793f1bb808191f97ff6b12`.
  Guard SHA-256:
  `daec5dcb4c7f5b3ea1ae19a9ac4d2990494b71fc8c5dccbe78f7f632167e7e74`.

The mandatory `bin/rtl-lint` / slang is unavailable. No independent signoff,
complete QST RTL regression, full synthesis/route, board or model test exists.
Source observations are **UNVERIFIED** beyond the tests and reports above.

## Review categories

1. Function: Q6 sampled vs reference; whole rows and all corner cases unproved.
2. Syntax: Icarus/Vivado accept; mandatory lint absent.
3. Synthesis: isolated OOC only; production default needs full-build proof.
4. Sequential: en gates the pipeline; guard valid/reset uses synchronous state.
5. Combinational: range checks and output mux; integration timing unmeasured.
6. Latches: no new procedural combinational block; lint unavailable.
7. FSM: guard is a shift pipeline, not formally checked for token alignment.
8. Case: no new production case; unused q4 in Q6 is deliberate.
9. Reset: selected in-flight reset test passed, external reset/drain unproved.
10. CDC: one-clock assumption, no CDC analysis.
11. Width: fixed delay, counters and port widths need mandatory lint.
12. Signedness: magnitude ordering assumes positive binary32 encodings.
13. Arithmetic: sampled FP32 arithmetic only; overflow inputs are rejected.
14. X/Z: unsupported and not asserted at integration.
15. Parameters: Q6 elaboration-time; guard hardcodes LM=2/LA=4/TW=2.
16. Protocol: en/valid/tag stalls sampled; no formal liveness or ownership.
17. Lint: required hook unavailable.
18. DFT: not assessed.
19. Power: not measured.
20. Maintainability: production default and experimental limitations documented.

Outstanding risks: global en/valid alignment in actual QST, accidental Q6/q4
misconfiguration, old-mode regressions outside tested patterns, reset while
writes are outstanding, exact model-scale semantics, actual SRAM/DDR hazards,
resource/timing effects, absence of independent and physical validation.
