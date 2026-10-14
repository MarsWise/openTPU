# Nextgen implementation progress

This ledger describes the work completed in this task run; it is not a claim that the complete
master brief has been delivered.

## Repository and provenance

- Upstream: `FeSens/openTPU`, default branch `main`, checked 2026-10-14.
- Verified upstream/base SHA: `5309a20d28948fa910897e85a96f82db6ee5c1fd`
  (commit date 2026-10-13 23:15:11 -03:00).
- Starting local HEAD matched that SHA; starting worktree was clean. Recovery branch:
  `backup/pre-nextgen-5309a20`.
- Configured origin: `https://github.com/MarsWise/openTPU.git`; authenticated account and write
  access are unverified because GitHub CLI (`gh`) is unavailable.
- Worktree: `C:\myOpenTPU\openTPU.worktrees\pasted-text-processing`.
- Integration branch: `feature/nextgen-hardware-first-vla-bitnet-vllm`.
- Verified implementation commit: `a97c516` (`quant: add ternary reference format`).
- No upstream merge or local-modification migration was needed at start.
- Publication: `NOT PUSHED`. The configured `origin` points to `MarsWise/openTPU`, but ownership,
  authenticated account, and write permission could not be verified (`gh` is unavailable).
  No draft PR was created.

## Subsystem status

| Subsystem | Status | Evidence / remaining work |
|---|---|---|
| Upstream provenance and baseline preservation | `REFERENCE VERIFIED` | Fetched `main` directly; fetched SHA equals starting HEAD; created recovery branch. Full 30/100-commit and merged-PR review remains incomplete. |
| Ternary symbol storage and software reference | `REFERENCE VERIFIED` | Added low-pair-first 2-bit pack/unpack and an int64 reference matmul; 22 focused direct assertions passed. Not a checkpoint quantizer or hardware mode. |
| Native ternary MXU / RTL / ISA / compiler / runtime | `PLANNED` | No RTL datapath, instruction, lowering, runtime dispatch, or real-checkpoint validation in this slice. |
| Matrix Engine 2.0 / adaptive precision | `PLANNED` | Existing matrix engine and precision paths were inspected; no RTL changes or DSE results. |
| YOLO/CNN, VLM, VLA, MoE changes, vLLM backend | `PLANNED` | No model/checkpoint was selected or run; no feature claims made. |
| `--strict-fpga` operator fallback audit | `PLANNED` | No strict mode added or verified. |
| Vivado / physical validation | `BLOCKED` | User explicitly owns all Vivado, bitstream, XDMA hardware, and board work. No such work was attempted. |

## Changes and interfaces

- `opentpu/quant.py`: generic ternary byte packing and exact software matmul reference.
- `tests/test_ternary.py`: round-trip, padding, malformed symbol, signed boundary and scalar
  reference checks.
- `docs/quant.md`: states encoding and explicitly disclaims MXU/checkpoint compatibility.
- `docs/nextgen/UPSTREAM_FEATURE_MATRIX.md`: scoped upstream evidence and audit limitations.
- No ISA encoding, RTL interface, board target, constraints, or resource profile changed.
- Encoding: `00=0`, `01=+1`, `10=-1`, `11=reserved`; four low-pair-first symbols per byte;
  unused final-byte pairs are zero. Dot-product accumulation reference is signed int64.
- No checkpoint/hash supplied or validated. “Tern2” remains undefined for a specific checkpoint.

## Verification record

- Python 3.12.10, NumPy 2.5.4, pytest 9.1.1. The test module's 22 assertions passed by direct
  invocation with deterministic seeds `0x7E2 + K` for K in 1, 3, 4, 5, 7, 8, 17.
- The direct check invoked all cases from `tests/test_ternary.py` without pytest collection:
  pack/unpack counts 0-9, row layout, reserved/padding rejection, type/shape guards, scalar
  matmul comparison and batched output shape.
- `python -m py_compile opentpu\quant.py tests\test_ternary.py`: PASS.
- `git diff --check`: PASS.
- `python -m pytest -q tests\test_ternary.py tests\test_quant.py`: BLOCKED at collection on
  Windows because `tests/conftest.py` imports `opentpu.rtlsim`, which imports POSIX-only `fcntl`.
  The initial attempts also identified missing pytest/NumPy; pytest 9.1.1 and NumPy 2.5.4 were
  installed in the user Python environment (no project manifest was changed).
- `Get-Command` found no `verilator` or `iverilog` executable; RTL simulation was NOT RUN.
- No model inference, performance measurement, synthesis estimate, Vivado run, or hardware test.
- No upstream baseline suite has been run in this task slice.

## Provisional resources and timing

- No RTL changes; no area, timing, traffic, cycles, or speedup estimates are produced.
- No physical results are available. User-reported Vivado utilization/timing must be recorded
  separately from simulations or estimates.

## User-only hardware handoff

No build-source or XDC change is included. Before any later user-owned implementation, preserve
the existing bitstream and reports, build from the documented board target, record Vivado version
and source revision, and share utilization, timing (WNS/TNS), bitstream, and board-test logs. Do
not program the FPGA until the user has reviewed and approved the build and rollback plan.
