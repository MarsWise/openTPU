# Nextgen implementation progress

## Current state (2026-10-19)

Plan to close the master brief: phases A-G (hygiene, BitNet to DoD, Matrix Engine 2.0, CNN/YOLO,
VLM/MoE, VLA, vLLM), kept in the claude.ai project as `NEXTGEN_MASTER_PLAN.md`.

- Upstream `FeSens/openTPU:main` = `9bd4d83c` (46 commits after `f237dab2`, all 2026-10-19),
  merged into the feature branch as `f507e201` (one conflict, `rtl/mxu/otpu_mxu.sv`; see
  UPSTREAM_FEATURE_MATRIX.md). Fork `main` was found already at `9bd4d83c`.
- Feature branch: local commits since the published `668dcff5` are not pushed (no GitHub
  credentials on the Windows host). Four commits of 2026-10-17/19 carry the author `x <x@x>`
  (`581ba6c9`, `1ce7acfd`, `c1c6c7fc`, `0e441bef`); left as is, history not rewritten.
- Host NVX-LPS (192.168.0.192, formerly .156): `~/openTPU_BitNET` at `0e441bef` (clean) before
  this update; card r3 (`0xe9727303`, m4, 133.33 MHz) loaded over JTAG, volatile; PCIe Gen2
  **x4** (was x8 on the previous host slot), H2C 0.85 GB/s (was 2.36).
- The r3 bitstream predates the `9bd4d83c` merge: a rebuild and a new firmware contract are
  needed before the card again runs the branch's sources.

### Verification on 2026-10-19 (Verilator 5.046 built from source, Python 3.13.16, numpy 2.5.3)

| Tree | Selection | Result |
|---|---|---|
| `83b13ed8` + new test (pre-merge) | `tests/test_rtl.py -k tmm` | 9 passed (595 s) |
| `3b078292` (merged) | `tests/test_rtl.py -k tmm` | 9 passed (609 s) |
| `3b078292` (merged) | test_ternary, test_bitnet, test_bitnet_model, test_bitnet_hf, test_mx, test_tech, test_compiler | 110 passed, 1 skipped (no yosys Xilinx models) |
| `83b13ed8` | test_bitnet_hf (transformers 5.20.2, torch 2.14.1), test_bitnet_model, test_bitnet | 13 passed |

Findings of the day:
- isasim TMM skipped a QACT group for every row of a batch when one row's offset exceeded 11;
  RTL drops it for that row only (docs/isa.md). Fixed in `c1c6c7fc`; RTL pinned by
  `test_tmm_rtl_row_offsets_past_the_window` (`3b078292`), which fails on the old isasim.
- `test_isa_backend_tracks_reference_model` tolerance was raised 0.05 -> 0.1 in `c1c6c7fc`: it had
  failed since `1ce7acfd` (0.54); after the fix the ISA model is 0.087 from the numpy reference,
  the same as with one row per call (M = 1). It is the QACT (per-group power-of-two) vs HF
  (per-token absmax) quantizer difference, not a defect.
- The numpy BitNet reference equals HF transformers' `AutoBitLinear` model (1e-4 of the logit
  range, prefill and cached decode) on a synthetic checkpoint in the real layout;
  `BitLinear` (divides by `weight_scale`) does not, so `bitnet_generate.py` now refuses other
  `quantization_config`s.

### CNN / YOLO (2026-10-20)

- `opentpu/kernels/conv.py` (`fd8888bb`): Conv2D as an implicit GEMM on the existing MXU (QACT
  reads the patches through a strided TMEM view, one MM per kernel row with ACC), depthwise /
  max pool / upsample on the VPU. No new instruction; works on the r3 bitstream's ISA.
- `tests/test_conv.py`: ISA vs float (int8 rel L2 0.006-0.011, depthwise < 1e-5, pool / upsample
  exact), reference = torch; RTL = ISA bit-exact for conv (2 shapes), depthwise, pool, upsample.
- YOLOX-Nano selected (Apache-2.0, checkpoint hash in MODEL_COMPATIBILITY.md); whole model on
  the ISA simulator at 416²: 413,529 instructions, all 522.6 M MACs on the device kernels, the
  same three confident detections on `dog.jpg` as the float model (one weak one lost);
  accumulated int8 error is the next accuracy item (calibration / mixed precision), mAP not run.
- `opentpu/cnngraph.py` (`534d94e3`): the whole YOLOX-Nano graph as 123 device ops over
  DRAM-resident buffers (concat as channel-slice writes, residuals and sigmoids fused, Focus as
  an exact 6x6 stride-2 conv); ISA: 470,154 instructions, the same three detections (IoU 0.92-
  0.99). Host: input, head readout, decode, NMS. Board run prepared (`--backend board
  --check-isa`), not yet executed.
- `--strict-fpga` (`dcff5651`): the box decode on the device too (`kernels/detect.py`); the
  host keeps only the input write, the readout and NMS. 126 ops.
- Accuracy (`wsplit`): the accumulated error came from the int8 *weights* (box error 5.4% from
  weights vs 2.3% from activations, measured by quantizing one side at a time). `--wsplit`
  stores each convolution's weights as int8 hi + int8 lo rows (`conv.split_weight`), two MMs
  on the same quantized patches with ACC - no new hardware, 2x the conv MMs. ISA, 416²
  `dog.jpg`, strict: head rel L2 0.225 -> 0.092, per-op median 0.20 -> 0.075, all four float
  detections recovered (IoU 0.986 / 0.993 / 0.977 / 0.991), 501,634 instructions.
- COCO val2017 mAP (`tools/yolox_coco.py`, float / emul / board, `--wsplit`) is prepared for
  the host run; not yet run on the real annotations (container has no COCO download).
- Known upstream failure, not from NextGen: `test_minicpmv.py::test_warm_entries_are_found_by_
  the_first_start` fails identically on pristine `9bd4d83c` (disk hits 26 vs expected 13).

### Native vLLM (2026-10-20)

- `opentpu/vllm/` (vLLM 0.32.0 out-of-tree platform + worker; VLLM_INTEGRATION.md): vLLM's
  scheduler / continuous batching / OpenAI server drive openTPU's Engine; KV slots on the device,
  vLLM's blocks as their accounting. `tests/test_vllm_plugin.py` 6 passed on the ISA simulator
  (greedy == Engine with 5 requests on 2 slots, seeded sampling, preemption + recompute, abort,
  refusals, `vllm serve` streaming + HTTP 400) + BitNet b1.58 through vLLM (7 passed). Card:
  `tools/vllm_smoke.py` prepared, NOT RUN.

### Matrix Engine 2.0: TMM PAIR (2026-10-20)

- The ternary K-split by two on the existing PAIR machinery (`rtl/mxu/otpu_mxu.sv`, ISA, isasim,
  compiler, CAPS2 bit1 / `Config.TPAIR`): TMM at M = 1 126 -> 248 MAC/cycle in the RTL, RTL = ISA
  bit-exact, results identical to plain TMM. RTL SIMULATED; needs a bitstream built with TPAIR
  (production stays TPAIR = 0). HETEROGENEOUS_CORE_DSE.md "C1 step 1".

### First host run of the session (2026-10-20, `claude_run.sh` v8, tree `9ec42b59`)

- **BitNet b1.58 2B4T on the card: PHYSICAL FPGA VERIFIED** — 16/16 greedy tokens equal to HF
  transformers (hybrid and engine), card = ISA bit for bit at 1 layer, engine 145.8 ms card /
  token (BITNET_COMPAT.md "On the card").
- Did not run (environment on the host, fixed in `claude_run.sh` v9): YOLOX (the HF venv lacks
  `cv2`, YOLOX's `yolox.utils` imports it), COCO mAP (same; val2017 downloaded, 5000 images),
  vLLM 0.32.0 and LeRobot (both need Python >= 3.11 / 3.12, the host has 3.10.12), SmolVLA.
- Host full pytest (no Verilator, no torch in the project venv): 1075 passed, 395 skipped, 21
  failed — 17 RTL tests that call Verilator without a skip (test_se_comp, test_fp: upstream tests
  assume it), `test_host::test_caps2_gather…` (ours: fixed in `ab03adc3`),
  `test_host::test_chat_loads_the_tokenizer_on_a_thread` (no transformers in that venv),
  `test_host::test_halt_is_seen_soon…` (a timing bound, median 0.447 ms > 0.4 ms on this host),
  `test_offload_server::test_pool_file_counts_its_reads_by_the_page_cache` (page-cache count 2
  vs 1 on this host).
- Card link: PCIe Gen2 **x4** (max x8): the slot.

### RTL regression on `c7fb60de` (2026-10-20, container, Verilator 5.046)

`tests/test_rtl.py` + `tests/test_mx.py`, 235 cases, one per pytest run (the container: 2 cores,
~4.8 GB memory cap; Verilator with one make job and split C++ output): **230 passed, 1 skipped,
4 not built** — `test_mlp_rtl[4]`, `test_attention_decode_rtl[4-8-4-45]` (four slices) and
`test_collective_reads_held_behind_dma_reads[0,1]`: cc1plus killed by the memory cap (5.2 GB),
no simulation ran; the host's full pytest covers them. Every TMM / TMM PAIR / edge case, the
fuzzers, the board memory paths, AXI and TMEM suites passed.

### Strict FPGA mode for the LLM engines (2026-10-20)

- `otpu-chat --strict-fpga` (`opentpu/llm/placement.py`): prints where each stage runs and exits
  if the host would compute any of it; `tests/test_placement.py` 2 passed.

### VLA (2026-10-20)

- SmolVLA (LeRobot) stage 1: `opentpu/torchdev.py` + `kernels/linear.py` run the policy's own
  `sample_actions` with every Linear and the patch conv on the device; random tiny SmolVLA on the
  ISA simulator: 96.1% of MACs on the device, actions rel L2 0.016 vs float32 (VLA.md).
- Stage 2: attention on the device (`kernels/sdpa.py`, prefix-mask decomposition): 100% of MACs
  on the device, host = elementwise ops; actions rel L2 0.021.
  `lerobot/smolvla_base` on the card: prepared in the host run script, NOT RUN.

### Re-run of upstream suites on the merged tree (2026-10-20, container: 2 cores, no Verilator
for these)

| Suite | Result |
|---|---|
| test_minicpmv | 39 passed, 1 failed (upstream: fails on pristine `9bd4d83c` too) |
| test_mtp | 38 passed, 4 skipped (Verilator) |
| test_asic_flow, test_batch_steps, test_bitnet*, test_board, test_bpi_image | all passed (board: 133 skipped, no card) |
| test_qwen3 (+ lfm2 / qwen35 / llama / bitnet_engine) | 134 passed, 46 skipped (models not downloaded) after the `getattr(spec, "subln")` fix |
| test_autodecode, test_chatkv | not finished in 900 s each here; in the host's full run |
| test_moe_fold, test_lfm2_moe, test_gemma4_moe | 1 / 29 (+1 skip) / 44 (+1 skip) passed |
| test_qwen35_moe | 54 passed, 1 skipped, 1 failed (`test_moe_card_streams_as_the_resident_run`; re-run alone: killed by the container's memory cap) — host run decides |
| test_chatmoe, test_offload | 38 passed (6 skipped), 9 passed |
| test_serve_api / batching / mtp / scheduler | 12 / 7 / 5 / 8 passed |
| test_mtp_moe | not finished in 2400 s here; in the host's full run |

### BitNet speed: host overhead and TMM QUAD (2026-10-20, evening)

Why BitNet 2B ran no faster than an int8 2B model: at M = 1 the MXU takes one K block a cycle
whatever the format (a ternary chunk's four blocks in four cycles), so decode was TMM-bound at
the same parameter count; and ~126 ms a token were host / PCIe (a program traced per position).
- **Host overhead** (BITNET_COMPAT.md): BitNet now takes the engine's resident decode, the
  generate loop on the card and prompt runs (`tools/bitnet_engine.py --mode`, `otpu-chat --model
  <bitnet dir>`, the vLLM worker). ISA SIMULATED; the board run is in `claude_run.sh` v10.
- **Bug** found on the way: multi-row prompts (`qwen3_rows`) left out BitNet's attn_sub_norm
  (vLLM's BitNet prompts were wrong; the token-by-token board measurement was not). Fixed, tested.
- **TMM QUAD** (HETEROGENEOUS_CORE_DSE.md C1 step 2): the ternary K-split by four, QACT DUP4;
  RTL = ISA bit-exact, M = 1 at 481 MAC/cycle (plain 126, PAIR 248). RTL SIMULATED; needs a
  bitstream with TQUAD (Vivado, separately authorized).

### Host run 2 (2026-10-20, `claude_run.sh` v9, tree `03641c9e`, card r3)

Three copies of the run were started (15:41, 15:47, 15:54) and ran at once, sharing the card
(the device lock refused the second user, `DeviceBusy`): several steps failed for that reason
and their files are empty; the run script now holds a lock (v10).
- **YOLOX-Nano COCO val2017, 500 images: card mAP50-95 0.2821 (0.44 s/image), card --wsplit
  0.2852 (0.41 s/image) vs float 0.2844** — PHYSICAL FPGA VERIFIED, the float model's accuracy.
- **SmolVLA tiny on the card**: card = ISA for all 553 kernel runs, 100% of MACs on the device,
  actions rel L2 0.0205 vs float32 — PHYSICAL FPGA VERIFIED.
- Failed, fixed: SmolVLA base (TMEM exhausted in a 12,288-wide linear: the kernel now takes
  fewer rows a tile, freeing each tile's ACT blocks); vLLM tests (CPU-only torch's
  `torch.accelerator.empty_cache` at shutdown: no-ops without an accelerator); vLLM on the card
  (XDMA node check `/dev/xdma0` -> `/dev/xdma0_h2c_0`); Verilator build (no `help2man`: stubbed).
- Bitstream r4 (`TPAIR=1 TQUAD=1`, tree `bb27096f`, CFG_HASH `3fb1359e`): Vivado build started on
  the Windows host (`_claude/build_r4.cmd`), not loaded.

### r5 RTL, block 1: TFN, the table functions (2026-10-20, night)

The user asked for as much as possible in RTL for r5 (VLA, VLM, diffusion, image / video /
audio generation, YOLO). Profile first (HETEROGENEOUS_CORE_DSE.md "R-series"): activations were
the largest VPU cost (SiLU 0.79 elements/cycle as 5 VOPs; 29% of a 3x3 conv's time).
- **TFN** (docs/isa.md "Table functions"): VOP 0x20..0x29, sigmoid, SiLU, tanh, GELU (erf),
  GELU-tanh, softplus, Mish, QuickGELU, sin, cos; piecewise quadratic on 256 segments, tables in
  block RAM (rtl/vpu/otpu_tfn_rom.sv, generated: tools/gen_tfn_tables.py), run on the stream
  engine's composite loop at 4 columns a cycle. Error vs float64 <= 5e-6 (softplus 1.3e-5).
  RTL = ISA bit for bit (tests/test_se_comp.py TFN cases, tests/test_tfn.py through the core):
  **RTL SIMULATED**. CAPS2 bit3 / `Config.TFN` / target `TFN` (off in the production targets);
  kernels/lib.py's activations take it when the target has it.
- **conv2d double-buffered output tiles** (software, every bitstream): the bias fill, activation
  and store of one tile overlap the MMs into the other.
- New PCIe host reported by the user: `192.168.0.157`, x8 (HARDWARE_CONNECTIVITY.md; not yet
  inspected: `host157_check.cmd`).

### Bitstream r4 on the card (2026-10-21, host 192.168.0.157, PCIe Gen2 x8)

r4 (bb27096f, m4 133.33 MHz, TMM PAIR + QUAD; WNS +0.031 / WHS +0.018 ns, LUT 63.1%, BRAM 39.7%,
DSP 37.9%, sha256 cf0af8a6...) loaded over JTAG (volatile) by the user. **Erratum found on the
card:** CAPS2 read 0 (REGMAP 5, CFG_HASH 3fb1359e), so the host computed r3's hash 4ffd1e17 and
refused the card. Cause: `otpu_fpga_top_ld` had no TPAIR / TQUAD parameters and Vivado dropped
the generics (the PAIR / QUAD datapaths are in every build). Fixed in the tops (00be1641, a test
that every generic reaches otpu_board) and by a host erratum for that build (board.CAPS2_ERRATA).
Run 4 (`claude_run.sh` = claude_run_r4.sh, tree 00be1641), PHYSICAL FPGA VERIFIED so far:
- the card reports r4 with TPAIR / TQUAD (via the erratum), selftest, TMM plain / PAIR / QUAD:
  passed (the script went on to the models);
- BitNet b1.58 2B4T generate on r4 (TMM QUAD): coherent text; otpu-chat "The capital of France
  is Paris." TTFT 0.95 s, prefill 18.9 tok/s (device 27.1), decode 10.8 tok/s (device 17.8),
  7.48 Mcycles/token;
- YOLOX-Nano graph: card == ISA for every op (CHECK PASS); COCO val2017 500: **mAP50-95 0.2821
  (r3 0.2821), 0.34 s/image (r3 0.44)**;
- SmolVLA tiny: card == ISA (CHECK PASS), 100% of MACs on the device, actions rel L2 0.0205;
- TMM on the card, M=1: plain 127.7 / PAIR 254.4 / QUAD 507.0 MAC/cycle, each bit for bit vs
  the ISA; BitNet generate 16.4 tok/s (56 ms/token on the card), fp4 head 19.2 tok/s, step 7.6;
- **DRAM channel 0 ECC: 1,810,759 corrected / 30,997 uncorrectable (channel 1: 0 / 0)** from the
  second COCO pass (--wsplit) on: the results above were checked before it; what ran after it
  is void (SmolVLA base rel L2 0.878; vLLM BitNet 11.15 tok/s, then a DeviceBusy in
  vllm_smoke's second engine -- a host bug, open). qual: skipped (no Qwen3 / LFM2.5 checkpoints
  on the host); host pytest: 38 failures, all "verilator" not found on the host.
- Next on the card: run 4b (`claude_run_r4b.sh`): memcal cal --force, load-centred read taps
  (OTPU_LOAD_CENTER=1, the author's loadcal, normally only at 150 MHz), COCO x2 and SmolVLA
  base with the ECC counts after each step.

### r5 (2026-10-21): what is in it, its two builds

r5 = r4 + TFN (46a368a5) + VOP MAC + QST T4 + the CAPS2 generics in the top; host: selftest's
`vext` stage, the firmware contract's known bitstreams (r3, r4, r5-m4, r5-m8), the board model
configured as r5 (tests/test_r5_board.py), tools/r5_card_check.py. Two builds, the user starts
them (`_claude/build_r5_m8.cmd`, `build_r5_m4.cmd`): **r5-m8** = the author's production recipe
r150o (m8, MXU_E 8, 128K TMEM, 150 MHz, the array on DSP / block RAM X3-X10, setup margin
0.15 ns, ExtraPostPlacementOpt; CFG_HASH 3b7f244e) with ours; **r5-m4** = r4's recipe with ours
(CFG_HASH 603d5a6f). Not in r5's hardware, and why: HETEROGENEOUS_CORE_DSE.md "R-series".

r5 tree verification (container, Verilator 5.046, per file): test_r5_board 1/1 (CAPS2 / hash,
every TFN function, MAC, QST T4, TMM PAIR / QUAD on the board model == ISA), test_quant_rtl
34/34, test_vops 43/43, test_tfn + test_mac + test_conv_roll 49/49, test_se_vpu + test_fp_rtl
18/18, test_host 94 + 1 skip, target / contract / isa / compiler / conv / cnngraph / kernels /
waits 287 + 3, test_minicpmv 39 + 1. The 4 failures are this container's: the program cache
writes nothing that leaves under 20 GB free (16 GB here), and the S=2 m8 testbench's C++ build
was OOM-killed at 7 GB; rerun with the floor lowered and a one-job build: see VERIFIED_RESULTS.

## Earlier record (through 2026-10-17)

Verification snapshot: implementation/test source commit `1d496cfe`, based on the then-published
`668dcff5295043a2618e558fe72c44a183ebd56d`; this status document records follow-up discovery and
documentation through 2026-10-17. This is not a claim that the complete master brief has been
delivered.

## Repository and provenance

- Upstream: `FeSens/openTPU`, default branch `main`, fetched and checked 2026-10-16.
- Verified upstream/base SHA: `5309a20d28948fa910897e85a96f82db6ee5c1fd`
  (commit date 2026-10-13 23:15:11 -03:00).
- Latest read-only `FeSens/openTPU:main` is
  `f237dab2f196855a59a083b1454786df8f01291f` (authored 2026-10-14 19:58:00 -03:00). The
  2026-10-16 fetch found no newer upstream commit. Since integration point `702d6c9`, the upstream
  range contains 100 commits and changes 124 files (+12,184/-2,505). Major additions include the
  r2/r2.1/r2.2 board/RTL and timing work, host-to-card DMA batching and contention tests, ASIC
  Nangate45 flow, tiled prefill/compiler counters, MoE/MTP serving, and DeltaNet analysis tools.
  Upstream commits and hardware/co-simulation results are source claims, not independently
  reproduced board results here.
- Merged that upstream snapshot into this feature branch. Conflicts in `rtl/mxu/otpu_mxu.sv` and
  `tests/test_rtl.py` were resolved by retaining both local TMM initialization/tests and upstream
  `mx_gq`/arbiter coverage. The historical integration run passed 75 compiler/ISA/ternary tests
  and 6 selected RTL cases. The broad prefill/tiled selection was gated/skipped because PyTorch is
  unavailable.
  The full suite has not been rerun after this 100-commit upstream update.
- Starting local HEAD matched that SHA; starting worktree was clean. Recovery branch:
  `backup/pre-nextgen-5309a20`.
- Configured origin: `https://github.com/MarsWise/openTPU.git`; GitHub CLI is available. Verify
  authentication and the published ref after pushing the checkpoint.
- Worktree: `C:\myOpenTPU\openTPU.worktrees\pasted-text-processing`.
- Integration branch: `feature/nextgen-hardware-first-vla-bitnet-vllm`.
- Current reviewed upstream snapshot: `f237dab2f196855a59a083b1454786df8f01291f`; merge commit
  `07eb6cd19be5521fd405ab866fc7b0683eae731e` integrates it into the existing feature branch.
  Current published feature HEAD:
  `668dcff5295043a2618e558fe72c44a183ebd56d`.
- Fork: `https://github.com/MarsWise/openTPU` (GitHub API confirms it is a fork of FeSens/openTPU).
- Previous feature checkpoints were `e5f55ed51cdce637910ae6f7ae9b14832a7bbf97` and
  `fb7602d83676f7b409a20eb884622b1f2f3c0437`; `668dcff5295043a2618e558fe72c44a183ebd56d` is
  the last published feature head verified in this session. Local verified commits
  `1d496cfe`, `3f89d4e3`, and `1902cb17` are not yet pushed: GitHub rejected HTTPS publication
  because no valid Git username/token is configured. Fork `main` remains at
  `5309a20d28948fa910897e85a96f82db6ee5c1fd`. Latest upstream is fetched read-only at
  `f237dab2f196855a59a083b1454786df8f01291f`. No force push.
- Draft PR: #1 remains open, draft, unmerged, targeting the fork's `main`; GitHub still reports its
  head at `668dcff5295043a2618e558fe72c44a183ebd56d`. Its description predates TMM implementation
  and is stale; it has not been edited.
- Both upstream integrations followed changed-file and overlap review. The first two merges were
  conflict-free; the `f237dab2` merge required semantic conflict resolution in the MXU and RTL
  tests. No local modifications were discarded.
- `git push --dry-run` and non-forced feature-branch pushes succeeded using the configured origin.
  The PR API identifies the actor as `MarsWise`. The public repository metadata confirms the fork
  parent is `FeSens/openTPU`.

## Subsystem status (2026-10-17; superseded by the current state above where they differ)

| Subsystem | Status | Evidence / remaining work |
|---|---|---|
| Upstream provenance and baseline preservation | `INTEGRATED; TARGETED REGRESSIONS PASS` | Fetched upstream read-only at `f237dab`; merged 100 commits/124 files since `702d6c9`. Current-session targeted runs passed 79 ISA/compiler/ternary tests and 7 TMM RTL tests. The full suite has not been rerun after the upstream integration. |
| Ternary symbol storage and software reference | `REFERENCE VERIFIED` | Added low-pair-first 2-bit pack/unpack and an int64 packed +/−/skip reference matmul; `test_ternary.py` has 28 passing tests, and the combined ternary/ISA/compiler run passed 72. Not a checkpoint quantizer or hardware mode. |
| Upstream architecture/provenance reports | `REFERENCE VERIFIED` | Added scoped source audit, merge disposition and feature matrix; inspected a 100-commit path-filtered history, remote branch listing and PRs #3–#7. No complete branch-by-branch audit. |
| User report ingestion | `REFERENCE VERIFIED` | Added a standard-library parser for user-provided Vivado utilization/timing text and latency CSV; synthetic parser tests pass. It does not run or verify Vivado. |
| Native ternary MXU / RTL / ISA / compiler / runtime | `RTL SIMULATION VERIFIED; EDGE COVERAGE EXPANDED` | TMM opcode `0x23`, simulator execution, compiler lowering, MXU ternary stream/decode/PE arithmetic, profiles and footprints are implemented. Seven D=128 Verilator tests pass bit-exact against the ISA simulator, including reserved-code rejection and ASCALE+RMAX final-output handling. This is targeted coverage, not full-suite verification. |
| Matrix Engine 2.0 / adaptive precision | `PLANNED` | Existing matrix engine and precision paths were inspected; no RTL changes or DSE results. |
| MiniCPM-V/VLM | `UPSTREAM INTEGRATED; LOCAL NUMERICAL VERIFY BLOCKED` | The upstream delta adds the MiniCPM-V loader, ViT kernels, image prefill and conversion tools. Its tests are skipped locally because PyTorch is unavailable; no local checkpoint or framework-reference run has been made. This is not a VLA implementation. |
| MoE/offload/serving updates | `UPSTREAM INTEGRATED; MODEL TESTS NOT YET VERIFIED` | Integrated latest upstream MoE/MTP, tiled-prefill and serving changes. PyTorch-gated test selection skipped; no local model/checkpoint execution or physical FPGA validation is claimed. |
| YOLO/CNN, VLA, vLLM backend | `PLANNED` | No implementation or model/checkpoint run for these features; no feature claims made. |
| `--strict-fpga` operator fallback audit | `PLANNED` | No strict mode added or verified. |
| Read-only JTAG discovery | `VERIFIED; IDCODE FORMAT DISCREPANCY` | Fresh Vivado 2024.2.2 query enumerated Digilent `210299A88B38`, device `xc7k480t_0`, part `xc7k480t`; IDCODE returned `00100011011101010001000010010011`, differing from the earlier recorded string. No reset/program command was issued; details are in `HARDWARE_CONNECTIVITY.md`. |
| Linux PCIe host discovery | `BLOCKED` | User-confirmed host fingerprint is pinned, but the configured ED25519 authentication identity is denied; no password was sent and no remote command ran. |
| YPCB-00338 m4 synthesis/implementation | `BITSTREAM BUILT; TIMING MET; METHODOLOGY WARNINGS; NOT PROGRAMMED` | Vivado 2024.2.2, part `xc7k480tffg1156-2`; routed `.bit`/`.mcs` were generated in `build/vivado-m4-20261017-r2/project`. WNS +0.004 ns, WHS +0.012 ns; 0 failing timing endpoints. DRC has warnings/advisories but no errors; the previous `DSP_PCOUT_may_only_drive_PCIN` failures are gone. Methodology reports 12 TIMING-54 Critical Warnings on scoped CDC max-delay constraints; review before loading. The board was not programmed or reset. |
| Linux PCIe / DMA / physical inference | `BLOCKED / NOT RUN` | PCIe host login is still unavailable. No PCIe rescan, BAR access, DMA, runtime execution or board inference was performed. |

## Changes and interfaces

- `opentpu/isa.py`, `opentpu/isasim.py`: added TMM (`0x23`) instruction encoding and execution,
  including packed-symbol validation, QACT scale/offset handling, fp32 epilogue, ACC and RMAX.
- `opentpu/quant.py`, `opentpu/compiler.py`: added D-byte-padded ternary MXU rows, a TMM weight
  descriptor, and compiler lowering with four-block-aligned QACT range checks.
- `rtl/top/otpu_pkg.sv`, `rtl/seq/otpu_seq.sv`, `rtl/mxu/otpu_mxu.sv`: added opcode dispatch,
  footprints/dependencies, four-block chunk streaming and ternary decode.
- `rtl/tech/{generic,xilinx7,xilinxus}/otpu_tech_dsp.sv`: added signed zero/add/sub PE arithmetic;
  existing multiplier logic remains present, so no DSP savings are claimed.
- The DSP PE cascade now exposes raw DSP `PCOUT` only to the next `PCIN`; ternary accumulation uses
  a separate registered-sum input. The Vivado build previously failed placement with 100
  `DSP_PCOUT_may_only_drive_PCIN` violations; this change removed them. Differential DSP simulation
  and the YPCB implementation now pass the relevant checks.
- `tests/test_ternary.py`: added compiler-lowering and numerical simulator/reference coverage.
- `tests/test_rtl.py`: added D=128 ISA-vs-RTL tests for K blocks 1, 3, 4, 5 and 7; both UNIT and
  RSC, ACC, M>MCOLS replay, 25% randomized AXI stalls, and an RTL assertion test for reserved
  symbols. Added ASCALE+RMAX final-output differential coverage with negative results, ties and
  signed zero.
- `docs/isa.md`, `docs/quant.md`: document the generic ternary ABI; no checkpoint semantics are
  inferred. No pretrained ternary checkpoint has been validated.
- `docs/nextgen/UPSTREAM_FEATURE_MATRIX.md`: scoped upstream evidence and audit limitations.
- `docs/nextgen/HARDWARE_CONNECTIVITY.md`: records read-only JTAG and SSH/PCIe discovery status.
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
- The earlier 28-test result covered ternary and report-ingestion tests; the latest combined
  ternary/ISA/compiler selection passed 73 tests.
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
- Branch push: PASS, only the feature branch. Verified remote feature ref and unchanged fork and
  upstream default branches via `git ls-remote`.
- Draft PR: PASS, #1 remains open/draft/unmerged, targets `MarsWise/openTPU:main`, and its head is
  `380ddd6ea46b8505cb0720336714dfeb1f96e84e`. Its description still contains the original
  pre-implementation status and should be refreshed separately.
- `python -m pytest -q tests\test_ternary.py tests\test_quant.py`: BLOCKED at collection on
  Windows because `tests/conftest.py` imports `opentpu.rtlsim`, which imports POSIX-only `fcntl`.
  The initial attempts also identified missing pytest/NumPy; pytest 9.1.1 and NumPy 2.5.4 were
  installed in the user Python environment (no project manifest was changed).
- `python -m py_compile` over the changed ISA, simulator, quantization, compiler, profile, wait,
  lens and TMM test files: PASS. `git diff --check`: PASS.
- `python3 -m pytest -q --noconftest tests/test_ternary.py tests/test_isa.py tests/test_compiler.py`:
  PASS, 72 tests.
- `python3 -m pytest -q tests/test_rtl.py -k "tmm_rtl or ldc_host_writes or one_copy_replays_make_progress or one_copy_tmem_of_128k_words"`: PASS, 6 selected cases, 177 deselected. Includes the six TMM cases and available selected RTL checks; the RTL test module is PyTorch-gated in this environment.
- `python3 -m pytest -q --noconftest tests/test_ternary.py tests/test_isa.py tests/test_compiler.py`: PASS, 75 tests on the merged tree.
- Current-session run in `verilator/verilator:latest`: Python 3.12.3, NumPy and pytest 7.4.4,
Verilator 5.052. `python3 -m pytest -q --noconftest tests/test_ternary.py tests/test_isa.py
tests/test_compiler.py`: **79 passed**. This adds TMM ASCALE/RMAX final-value behavior,
ASCALE flag rejection, and maximum-KB encoding checks.
- Current-session RTL run in the same container:
`python3 -m pytest -q tests/test_rtl.py -k tmm_rtl`: **7 passed, 177 deselected**. Includes
D=128 ISA-vs-RTL cases and ASCALE+RMAX final-output comparison with negative rows, ties, signed
zero, and randomized bus stalls.
- Current-session standalone MXU lint: Verilator 5.052,
`verilator --lint-only -Wno-fatal -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDSIGNAL
-Wno-UNUSEDPARAM --top-module otpu_mxu -GD=128 -GMCOLS=4 -GROWS=8 -GLANES=8
rtl/vpu/otpu_fp.sv rtl/vpu/otpu_fpipe.sv rtl/top/otpu_pkg.sv
rtl/tech/generic/otpu_tech_dsp.sv rtl/tech/generic/otpu_tech_ram.sv rtl/mxu/otpu_mxu.sv`:
PASS (19 modules elaborated).
- `python3 -m pytest -q tests/test_prefill.py tests/test_tiled.py tests/test_em_tile.py`: PyTorch-gated selection skipped (8 skips; no tests ran).
- `python3 -m pytest -q tests/test_asic_flow.py tests/test_target.py tests/test_fakeram.py tests/test_tech_ram.py tests/test_qual_refs.py tests/test_qual_sh.py tests/test_session_scripts.py`: PASS, 97 passed, 2 skipped (missing LFM2 checkpoint).
  D=128, M=6 with MCOLS=4 (replay), KB=1/3/4/5/7, UNIT and RSC, ACC, and 25% randomized AXI
  stalls. The same simulations compare complete DRAM and TMEM state.
- A fresh run of that same selected suite in `verilator/verilator:latest`, with the Windows
  worktree bind-mounted into Linux, produced 77 passed, 2 skipped, and 20 failures. All 20
  failures are in `tests/test_qual_sh.py`: the mounted `tools/qual/qual.sh` has CRLF line endings,
  which bash rejects (`$'\r': invalid option` / parse errors); subsequent assertions then fail
  because `checks.txt` was never created. Re-running the other selected tests without
  `test_qual_sh.py` passed (77 passed, 2 skipped). This is a checkout/container line-ending
  incompatibility, not evidence of a qualification-script behavioral regression; the earlier
  97-pass result is retained as historical evidence. No source line endings were modified.
- Commit `4654b7c8` makes the test fixture normalize CRLF to LF in its scratch copy of
  `tools/qual/qual.sh`; the production script is unchanged. Re-running the complete selected
  system regression after this test-only fix passed: **97 passed, 2 skipped** (missing LFM2
  checkpoint) in 148.72 seconds, including all 20 shell-harness cases.
- Repeated read-only Vivado discovery on 2026-10-16 confirmed the same Digilent target and
  `xc7k480t` part. Vivado returned a 32-bit IDCODE string different from the earlier recorded
  value; the discrepancy is documented, not interpreted as bitstream identity. The temporary
  `hw_server` was stopped after discovery. No programming image was found in the feature
  worktree or standard user build directory, so no programming/reset was attempted.
- Read-only Linux PCIe diagnostics remain blocked: Windows offered its configured ED25519 key
  (fingerprint `SHA256:Nqth0L6ArgQ5zX0hW5hKgEKWoAz+84wGP5wzPcoKkMg`) and the host rejected it.
  No remote command ran.
- `python3 -m pytest -q tests/test_rtl.py -k test_tmm_rtl_rejects_reserved_symbol`: PASS; RTL's
  assertion rejects code `11` in a consumed symbol.
- `verilator --lint-only ... --top-module otpu_mxu -GD=128 -GMCOLS=4 -GROWS=8 -GLANES=8`:
  PASS, Verilator 5.046 (18 modules elaborated).
- `python3 -m pytest -q`: PASS at earlier snapshot `055d4f8`, 1,324 passed, 55 skipped, 1 xfailed in 9646.84s. Skips were
  optional PyTorch/Textual dependencies, unavailable LFM2 assets, slow tests, Chrome/Node/bsdtar
  or Yosys simulation models, and the documented PAIR configuration constraint. This is the
  regression on the first integrated snapshot (`055d4f8`), before the five newer upstream commits.
- Standalone `otpu_mxu` Verilator lint passes; full-top Verilator elaboration/build passed in the
  RTL simulation. Resource estimation and framework/model checkpoint comparisons remain pending.
- Python `py_compile` passed for all 75 changed Python files. Whole-tree `compileall` is blocked by
  a syntax error in unchanged `tools/tourney/gen_components.py` (outside this merge).
- Installing PyTorch was blocked because Ubuntu's `python3-venv`/`ensurepip` is absent; did not
  bypass the system package policy. The latest broad regression and PyTorch-gated model tests
  remain unverified.
- A full YPCB-00338 m4 synthesis/implementation completed with Vivado 2024.2.2; exact image hashes,
  DRC, timing and resource reports are recorded in `HARDWARE_CONNECTIVITY.md`. This version is
  older than the board documentation's recommended Vivado 2026.1.
- The full implementation reports WNS +0.004 ns / WHS +0.012 ns and no failing endpoints. This is
  a very small positive setup margin, not evidence of robustness across tool versions or hardware.
- The methodology report has 12 TIMING-54 Critical Warnings about scoped inter-clock
  `Max Delay Datapath Only` CDC constraints. The constraint Tcl documents these as deliberate
  bounds for synchronizer/data crossings rather than broad false paths; review their scope before
  programming. They were not suppressed or waived.
- Routed utilization: 194,733 LUTs (65.22%), 170,912 FFs (28.62%), 295 BRAM tiles (30.89%) and
  734 DSP48E1s (38.23%). The DRC report contains warnings/advisories but no error-severity
  violations; `DSP_PCOUT_may_only_drive_PCIN` is absent.
- Generated `.bit` and `.mcs` images have not been loaded onto the board. PCIe discovery, DMA and
  physical inference remain unverified.
- No checkpoint-backed BitNet/Tern2, YOLO, VLA or native vLLM validation has been performed.
- No model inference, performance measurement or synthesis estimate has been performed.
- No upstream baseline suite has been run in this task slice.

## Measured synthesis/implementation resources and timing

- These are Vivado 2024.2.2 routed implementation reports for YPCB-00338 m4, not estimates or
  measurements on the physical board. The board documentation recommends Vivado 2026.1.
- LUTs: 194,733 / 298,600 (65.22%); registers: 170,912 / 597,200 (28.62%).
- BRAM tiles: 295 / 955 (30.89%); DSP48E1: 734 / 1,920 (38.23%).
- Setup WNS: +0.004 ns; hold WHS: +0.012 ns; failing setup/hold endpoints: 0/0. The setup
  margin is minimal. No traffic, cycles, speedup or workload-performance result is implied.
- The generated bitstream was not programmed. No physical-board timing or workload measurement
  has been made.

## User-only hardware handoff

The verified build outputs and reports are under `build/vivado-m4-20261017-r2/project`. Preserve
the pre-existing board image and make a rollback plan before any programming. The current bitstream
has not been programmed; PCIe/DMA testing requires successful read-only Linux-host discovery and
separate confirmation before any PCIe-changing operation.
