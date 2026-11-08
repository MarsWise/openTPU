# Experimental 16-value INT6 word formatter

Status: **MANUAL_REVIEW_REQUIRED**. This is an isolated prototype; it is not
instantiated by QST, firmware, or the memory controller. No FPGA support or
performance claim follows from these tests.

## Contract

On `in_valid_i && in_ready_o`, accept 16 signed 8-bit lanes in
`values_i[8*i +: 8]`. Each lane must be in [-32,31]. Code `i` occupies
bits `[6*i +: 6]`; emit three words in ascending address order, each with
`be_o=4'b1111`. Address is a byte address, word-aligned and <= `0xfffffff4`.
The output word address is byte address divided by four. No tails or unaligned
inputs are supported. Invalid input or address produces a one-cycle reject,
no write, and no reject backpressure. Outputs are held stable under stalls.
An unrelated input offered while busy is not accepted. The input and output
share one clock. `rst` is synchronous for state and gates valid/ready outputs;
reset must be stable and synchronous at the interface.

An accepted output is **not a memory completion**. Three writes are not atomic;
reset can leave earlier words accepted, pending or committed. The integrating
design must capture reject, retain metadata, track write acknowledgements,
fence data and scale dependencies, prevent readers from seeing partial data,
and define reset/recovery before this block is considered for QST.
The format is not a complete quantizer, numerical model, INT6 KV or MXU read
path. Unknown input bits are outside the binary-input contract.

## Evidence

* Red test: compilation failed for absent module, exit 2, recorded in
  `build/logs/kv_int6_word_red.log`.
* Icarus `-g2012 -Wall` passes 256 generated groups, stalled payload,
  competing input, 64-byte/4-KiB/4-GiB boundaries, unaligned and overflow
  rejection, both out-of-range directions for each of 16 lanes, byte enables,
  and a reset after one accepted word. Output:
  `build/logs/kv_int6_word_test.log`. One earlier testbench run failed a
  same-delta reset check; the test now waits 1 ns after deasserting reset before
  checking combinational ready. This correction does not prove reset behavior
  for an external asynchronous source.
* Vivado 2024.2 isolated out-of-context synthesis completed with exit 0 on
  `xc7k480tffg1156-2`: 135 LUT, 130 registers, 0 BRAM, 0 DSP. Reports are in
  `build/logs/kv_int6_word_ooc_*.rpt`; CFGBVS-1 appears in the OOC DRC.
  This is post-synthesis only, not placed/routed or a full-design timing result.
  Stream SHA-256: `c7d19b891c0d45edbe2dc62753f1ea0fa07c6f8715eac3f4a0395542985afc3f`.
* Mandatory `bin/rtl-lint` and slang are absent; no `[slang]` results exist.
  No independent review, formal proof, CDC analysis, physical validation,
  real memory protocol simulation, or full r8 implementation has been done.
* For the codec submodule, all 16,777,216 packed words and 1,024 signed-range
  cases passed independently; see `REVIEW_STATUS.md`. This does not exhaust
  all possible stream sequences.

## Review categories (source observations are reasoning, not lint proof)

1. Functional: packed values, addresses and masks tested; integration missing.
2. Syntax: Icarus elaboration succeeds; required lint unavailable.
3. Synthesis: OOC passed; full r8 placement/routing remains unchecked.
4. Sequential: one rising-edge state block; nonblocking assignments.
5. Combinational: no output-ready to input-ready combinational path.
6. Latches: continuous combinational assignments; mandatory lint absent.
7. FSM: idle/busy and 0..2 counter; not formally proved.
8. Case: none in synthesizable RTL.
9. Reset: synchronous state reset; partial writes are not rolled back.
10. CDC: single-clock contract; crossing or asynchronous reset requires review.
11. Width: counter, address and packed-slice intent checked; no width lint.
12. Signedness: codec checks 8-bit input encodings; no signedness lint.
13. Arithmetic: alignment and range guard protect three output words from wrap.
14. X/Z: unsupported inputs require integration assertions.
15. Parameter: no RTL parameters.
16. Protocol: directed stalled and competing-input scenarios passed; no proof.
17. Lint: mandatory hook and slang absent.
18. DFT: no scan or design-for-test validation.
19. Power: no activity or power analysis.
20. Maintainability: limitations are explicit; no production integration.

Outstanding risks: full-elaboration latch/width/signedness lint; interface CDC;
reset during outstanding writes; unknown signals; missing write completion and
atomicity; one-cycle rejection; system deadlock and scoreboard correctness;
independent protocol proof; memory/model numerical accuracy; physical timing
and implementation. These are blockers to signoff and integration, not optional
optimizations.
