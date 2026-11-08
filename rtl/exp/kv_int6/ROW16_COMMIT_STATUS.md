# Experimental INT6 row and scale commit

Update: the commit module now propagates `AUTO_AMAX` (default 0) to its row
writer. See `ROW16_AUTO_STATUS.md` for current evidence; previous OOC figures
and source hashes below describe the pre-parameter version.

Status: **MANUAL_REVIEW_REQUIRED**. This is a standalone experiment, not
connected to the production QST, ISA, DDR controller, MXU or r8 bitstream.
Do not interpret its tests or out-of-context synthesis as model support.

`otpu_qst6_row16_commit` accepts sixteen FP32 values and a caller-supplied
amax. It writes three packed INT6 words, waits for the supplied global
write-idle indication, writes a separate FP32 scale word and waits for idle
again before `done`. It rejects unaligned addresses, a data range crossing
the 32-bit boundary, or scale overlapping the three data words. The child
checks amax and lanes for supported finite inputs. Reset enters recovery and
blocks new transactions until write-idle is asserted.

The caller must ensure amax matches the row, the allocation lies in actual
available memory, the global idle indication includes all external buffered
writes and the corresponding completion, and ownership prevents overlapping
accesses. The transaction is **not atomic**: reset cannot undo accepted
writes, and interrupted rows must be invalidated by the caller. There is no
memory error response, timeout, cross-port/host scoreboard, real memory
controller test, automatic amax reduction, INT6 MXU reader or V transpose.
Untrusted input validation is limited to the documented interface checks.

The Icarus test uses a simulated ready/idle endpoint, not a real DDR controller.
It compares 12 data/scale groups against the Python reference, and exercises
stalls, stable writes, both drain fences, address boundaries and overlaps,
invalid inputs, and reset with outstanding writes. A second test uses a
behavioral model of the slice SW register and a delayed single-entry memory
queue, checking final memory contents and both completion fences; it does not
instantiate the actual slice or DDR controller. The independent reviewer
must examine handshakes, valid/ready stability, address bounds, buffer
ownership, reset, idle semantics, arithmetic corner cases, parameter widths,
CDC assumptions and existing QST regression before integration.

The required `bin/rtl-lint`/slang review is unavailable in this environment;
simulation and synthesis do not replace it. OOC synthesis is a whole-block
estimate and cannot establish incremental r8 resource use or timing closure.

Vivado 2024.2.2 OOC synthesis, XC7K480T-2, 7.500 ns clock, final module SHA256
`11685eb12d791ca7367ffd47612e1fc44179636b605f5ded49b2510d83bf5dd3`:
3056 LUT, 3480 FF, 18 DSP and 0 BRAM, including all child datapath logic.
No placement or routing was run. The DRC has CFGBVS-1 and DSP pipelining
warnings; `HD.CLK_SRC` is unset, so OOC timing does not establish closure.
The separate combined selection returned 60 passed and 14 deselected; it is
not a full r8 regression. Reports and reproducible OOC TCL are under `build/`
and are ignored build artifacts, not part of a committed release.
