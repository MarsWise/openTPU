# Experimental two-pass INT6 row controller

Status: **MANUAL_REVIEW_REQUIRED**. `otpu_qst6_row_twopass` is standalone
experimental RTL. It is not connected to the production QST, TMEM, DDR, ISA,
MXU, host runtime or any bitstream; its tests do not establish model support.

For 1..255 groups of 16 values, the first pass captures one group at a time,
checks each lane for NaN/infinity, flushes subnormal magnitudes to zero and
computes the maximum across the whole row. The second pass invokes the existing
row writer with this maximum; the source must replay **exactly the same**
groups in order, with `valid` and values held until `ready`. One common scale
and three packed 32-bit data words per group are written. Completion is based
on the child's two externally supplied global-idle fences.

This removes the need for the *caller to calculate* the row maximum for this
prototype; it does not create a hardware TMEM source or numerical FPGA-only
model execution. The interface does not retain or compare the entire row:
external ownership must prevent source changes between the passes. It checks
alignment, 32-bit range and data/scale overlap, but not the actual DDR capacity.
Memory errors and timeout are not reported. Writes are not atomic; after reset
or a failed replay, the owner must invalidate any partially written row. Reset
cannot cancel writes and recovery relies on truthful shared `wr_idle`.
The controller does not support a partial 16-value group or V-cache transpose.

## Evidence

* Red tests first failed because the module did not exist. The first post-code
  run had two testbench errors: it expected command-ready too early after done;
  it also had one Icarus-generated VVP `Invalid opcode`/syntax failure. The
  test was corrected to check completion before waiting for the next ready;
  the Icarus cause remains unestablished.
* The final selected series: **20 passed** including the five new cases and
  existing shared-row, decoder and ISA bridge tests. An earlier combined
  selection reported **19 passed, 1 failed**, the failure being Icarus
  `0xC0000005` in an existing auto-amax test. Its repeated combined selection
  passed; this does not establish the cause or eliminate tool instability.
* New tests compare packed words and scale with the Python INT6 reference for
  1, 2, 17 and 255 groups, including a maximum in the final group; zero and
  negative subnormal; NaN and infinity in the last lane of a later scan group;
  invalid command boundaries; stalled output stability; delayed idle; reset
  during scan and after an accepted write. The memory is a behavioral model;
  there is no real DDR or complete model test.
* Vivado 2024.2.2 OOC synthesis for XC7K480T-2 at 7.500 ns completed, exit 0:
  **3384 LUT, 3561 FF, 18 DSP, 0 BRAM** for the entire hierarchy, including
  the child writer. OOC WNS +1.618 ns / WHS +0.123 ns are pre-placement and
  use incomplete external constraints with `HD.CLK_SRC` unset. They do not
  establish timing or incremental resources of the complete r8. DRC includes
  the normal OOC configuration and DSP pipeline warnings. Reports are under
  `build/logs/qst6_row_twopass_ooc_*` and are ignored build artifacts.
* RTL SHA-256: `80ebb481d8b37db9dae8f17e0778999b8f0e53c815b86b1011b5ce60edb77167`.

Mandatory `rtl-review-signoff` lint (`bin/rtl-lint`/slang) is unavailable;
no independent review, formal protocol proof, actual memory controller test,
CDC analysis, complete r8 regression, routed build, model comparison or
physical test has been performed. In particular, the replay ownership and
reset/drain contracts require validation before any production integration.
