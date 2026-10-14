# Nextgen implementation progress

This ledger describes the work completed in this task run; it is not a claim that the complete
master brief has been delivered.

## Repository and provenance

- Upstream: `FeSens/openTPU`, default branch `main`, checked 2026-10-14.
- Verified upstream/base SHA: `5309a20d28948fa910897e85a96f82db6ee5c1fd`
  (commit date 2026-10-13 23:15:11 -03:00).
- Latest read-only `FeSens/openTPU:main` fetched 2026-10-14 as
  `e15b4a44234661df7e14176b90669ead4d579fbe`, 51 commits beyond the recorded base (55 files,
  +7467/-243). No upstream files or refs were written.
- The upstream delta adds MiniCPM-V/VLM loading, ViT kernels, serving/offload and model tools;
  updates MoE/Qwen3 behavior; and makes the SW output register unconditional across targets.
  `opentpu/compiler.py` and `opentpu/llm/qwen3.py` overlap with local TMM edits, but their reviewed
  upstream hunks are in separate areas. Integration is pending until the in-flight regression
  finishes, so its source tree remains stable.
- Starting local HEAD matched that SHA; starting worktree was clean. Recovery branch:
  `backup/pre-nextgen-5309a20`.
- Configured origin: `https://github.com/MarsWise/openTPU.git`; authenticated account and write
  access are unverified because GitHub CLI (`gh`) is unavailable.
- Worktree: `C:\myOpenTPU\openTPU.worktrees\pasted-text-processing`.
- Integration branch: `feature/nextgen-hardware-first-vla-bitnet-vllm`.
- Current code checkpoint: `92f7e405b9b2018c85c4f420d5f333d798cc1a2b`
  (`tools: ingest user-supplied FPGA reports`). Latest pushed normalization/docs checkpoint:
  `23bef65011f9cc4a39b64b6f08ab3f8f47f95d82` (includes the PR status refresh).
- Fork: `https://github.com/MarsWise/openTPU` (GitHub API confirms it is a fork of FeSens/openTPU).
- Branch push verified at the latest GitHub check: remote feature ref resolves to
  `23bef65011f9cc4a39b64b6f08ab3f8f47f95d82`; remote `main` remains
  `5309a20d28948fa910897e85a96f82db6ee5c1fd`. No force push.
- Draft PR: created and verified as open/draft #1:
  `https://github.com/MarsWise/openTPU/pull/1`, targeting the fork's `main`.
- No upstream merge or local-modification migration was needed at start; latest upstream
  integration is now pending the review and regression above.
- `git push --dry-run` and non-forced feature-branch pushes succeeded using the configured origin.
  The PR API identifies the actor as `MarsWise`. The public repository metadata confirms the fork
  parent is `FeSens/openTPU`.

## Subsystem status

| Subsystem | Status | Evidence / remaining work |
|---|---|---|
| Upstream provenance and baseline preservation | `REVIEWED; INTEGRATION PENDING` | Fetched `upstream/main` read-only at `e15b4a4`; reviewed the 51-commit/55-file summary, compiler/Qwen3 overlap, simulation and RTL-top changes. Full patch-level review remains incomplete; merge is deferred until the current regression finishes. |
| Ternary symbol storage and software reference | `REFERENCE VERIFIED` | Added low-pair-first 2-bit pack/unpack and an int64 packed +/−/skip reference matmul; `test_ternary.py` has 28 passing tests, and the combined ternary/ISA/compiler run passed 72. Not a checkpoint quantizer or hardware mode. |
| Upstream architecture/provenance reports | `REFERENCE VERIFIED` | Added scoped source audit, merge disposition and feature matrix; inspected a 100-commit path-filtered history, remote branch listing and PRs #3–#7. No complete branch-by-branch audit. |
| User report ingestion | `REFERENCE VERIFIED` | Added a standard-library parser for user-provided Vivado utilization/timing text and latency CSV; synthetic parser tests pass. It does not run or verify Vivado. |
| Native ternary MXU / RTL / ISA / compiler / runtime | `RTL SIMULATION VERIFIED` | TMM opcode `0x23`, simulator execution, compiler lowering, MXU ternary stream/decode/PE arithmetic, profiles and footprints are implemented. The current `test_tmm_rtl` selection passed all 6 D=128 Verilator cases, bit-exact against the ISA simulator and including invalid-code rejection. The full suite was interrupted after an environment failure (`transformers` missing); the exact host test passed after installing that test dependency, and a full run remains pending. |
| Matrix Engine 2.0 / adaptive precision | `PLANNED` | Existing matrix engine and precision paths were inspected; no RTL changes or DSE results. |
| YOLO/CNN, VLM, VLA, MoE changes, vLLM backend | `PLANNED` | No model/checkpoint was selected or run; no feature claims made. |
| `--strict-fpga` operator fallback audit | `PLANNED` | No strict mode added or verified. |
| Vivado / physical validation | `BLOCKED` | User explicitly owns all Vivado, bitstream, XDMA hardware, and board work. No such work was attempted. |

## Changes and interfaces

- `opentpu/isa.py`, `opentpu/isasim.py`: added TMM (`0x23`) instruction encoding and execution,
  including packed-symbol validation, QACT scale/offset handling, fp32 epilogue, ACC and RMAX.
- `opentpu/quant.py`, `opentpu/compiler.py`: added D-byte-padded ternary MXU rows, a TMM weight
  descriptor, and compiler lowering with four-block-aligned QACT range checks.
- `rtl/top/otpu_pkg.sv`, `rtl/seq/otpu_seq.sv`, `rtl/mxu/otpu_mxu.sv`: added opcode dispatch,
  footprints/dependencies, four-block chunk streaming and ternary decode.
- `rtl/tech/{generic,xilinx7,xilinxus}/otpu_tech_dsp.sv`: added signed zero/add/sub PE arithmetic;
  existing multiplier logic remains present, so no DSP savings are claimed.
- `tests/test_ternary.py`: added compiler-lowering and numerical simulator/reference coverage.
- `tests/test_rtl.py`: added D=128 ISA-vs-RTL tests for K blocks 1, 3, 4, 5 and 7; both UNIT and
  RSC, ACC, M>MCOLS replay, 25% randomized AXI stalls, and an RTL assertion test for reserved
  symbols.
- `docs/isa.md`, `docs/quant.md`: document the generic ternary ABI; no checkpoint semantics are
  inferred. No pretrained ternary checkpoint has been validated.
- `docs/nextgen/UPSTREAM_FEATURE_MATRIX.md`: scoped upstream evidence and audit limitations.
- `docs/nextgen/ARCHITECTURE_AUDIT.md`, `UPSTREAM_MERGE_PLAN.md`, `CORE_DECISIONS.md`,
  `RESOURCE_BUDGET.md`, `MODEL_COMPATIBILITY.md`, `OPERATOR_COVERAGE_MATRIX.md`,
  `HETEROGENEOUS_CORE_DSE.md`, `VLLM_INTEGRATION.md`, and `VERIFIED_RESULTS.csv` record audited
  facts, completed software tests, and explicit unimplemented/NOT RUN status.
- `tools/nextgen/ingest_reports.py` parses supplied report snippets and latency CSV into a
  provenance-labeled JSON record; `tests/test_nextgen_ingest.py` checks parsing and rejection
  cases. It is not a substitute for Vivado or user report review.
- No ISA encoding, RTL interface, board target, constraints, or resource profile changed.
- ABI: `00=0`, `01=+1`, `10=-1`, `11=reserved`; four low-pair-first symbols per byte. Four
  consecutive D-element K blocks share one D-byte chunk; partial chunks are padded with zero.
  TMM uses fixed `W=11`, `m=1`, QACT scales/offsets, optional per-row fp32 RSC scale, and requires
  exactly one of UNIT/RSC. The packed software reference accumulates signed integer products.
- No checkpoint/hash supplied or validated. “Tern2” remains undefined for a specific checkpoint.

## Verification record

- Python 3.12.10, NumPy 2.5.4, pytest 9.1.1. The ternary module passed 23 tests under pytest
  using a temporary copy outside the repository test tree to avoid `tests/conftest.py`'s POSIX
  import:
  `python -m pytest -q --confcutdir=<temp-test-dir> <temp-test-dir>\test_ternary.py`
  Result: `23 passed in 0.10s`.
- `python -m pytest -q --noconftest tests\test_ternary.py tests\test_nextgen_ingest.py`:
  PASS, 28 tests. `--noconftest` avoids the suite's POSIX-only `fcntl` import; this run includes
  both the packed ternary checks and the synthetic report-ingestion tests.
- The 28-test result was repeated after code commit `92f7e405b9b2018c85c4f420d5f333d798cc1a2b`.
- Verilator 5.020 was too old for this full-top build and failed while elaborating the existing,
  unchanged stream-tail module. Built Verilator 5.046 from the official `v5.046` tag
  (`24b2ac24c721fdad89bba75a492e02c6aa63f32e`) in WSL; full-top compile/elaboration then passed
  for the TMM configurations.
- The tests cover pack/unpack counts 0-9, row layout, reserved-code/non-zero-tail rejection in
  decode and matmul, type/shape guards, scalar matmul comparison, and batched output shape.
  Matmul reference comparisons use deterministic seeds `0x7E2 + K` for K in 1, 3, 4, 5, 7, 8,
  17.
- `python -m py_compile opentpu\quant.py tests\test_ternary.py`: PASS.
- `git diff --check`: PASS.
- Branch push: PASS, only the feature branch. Verified remote feature ref and unchanged default
  branch via `git ls-remote`.
- Draft PR: PASS, #1 is open, marked draft, targets `MarsWise/openTPU:main`, and its head is
  `23bef65011f9cc4a39b64b6f08ab3f8f47f95d82`. The PR description was refreshed through GitHub's
  REST API after adding report-ingestion tooling.
- `python -m pytest -q tests\test_ternary.py tests\test_quant.py`: BLOCKED at collection on
  Windows because `tests/conftest.py` imports `opentpu.rtlsim`, which imports POSIX-only `fcntl`.
  The initial attempts also identified missing pytest/NumPy; pytest 9.1.1 and NumPy 2.5.4 were
  installed in the user Python environment (no project manifest was changed).
- `python -m py_compile` over the changed ISA, simulator, quantization, compiler, profile, wait,
  lens and TMM test files: PASS. `git diff --check`: PASS.
- `python3 -m pytest -q --noconftest tests/test_ternary.py tests/test_isa.py tests/test_compiler.py`:
  PASS, 72 tests.
- `python3 -m pytest -q tests/test_rtl.py -k test_tmm_rtl`: PASS, 5 bit-exact ISA-vs-RTL cases.
  D=128, M=6 with MCOLS=4 (replay), KB=1/3/4/5/7, UNIT and RSC, ACC, and 25% randomized AXI
  stalls. The same simulations compare complete DRAM and TMEM state.
- `python3 -m pytest -q tests/test_rtl.py -k test_tmm_rtl_rejects_reserved_symbol`: PASS; RTL's
  assertion rejects code `11` in a consumed symbol.
- `verilator --lint-only ... --top-module otpu_mxu -GD=128 -GMCOLS=4 -GROWS=8 -GLANES=8`:
  PASS, Verilator 5.046 (18 modules elaborated).
- The broad openTPU regression suite is in progress. Resource estimation and framework/model
  checkpoint comparisons remain pending.
- No model inference, performance measurement, synthesis estimate, Vivado run, or hardware test.
- No upstream baseline suite has been run in this task slice.

## Provisional resources and timing

- No synthesis/resource estimate, area, timing, traffic, cycles, or speedup claim is produced.
- No physical results are available. User-reported Vivado utilization/timing must be recorded
  separately from simulations or estimates.

## User-only hardware handoff

No build-source or XDC change is included. Before any later user-owned implementation, preserve
the existing bitstream and reports, build from the documented board target, record Vivado version
and source revision, and share utilization, timing (WNS/TNS), bitstream, and board-test logs. Do
not program the FPGA until the user has reviewed and approved the build and rollback plan.
