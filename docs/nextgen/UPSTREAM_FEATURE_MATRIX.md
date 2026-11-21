# Upstream feature and compatibility matrix

## 2026-10-19: `f237dab2..9bd4d83c` (46 commits, merged in `f507e201`)

- `FeSens/openTPU:main` = `9bd4d83c5e03231057f9a9eed22370747ec89d0a`, committed 2026-10-19
  11:02:12 -03:00 ("Merge timing-ffs (d3fa1105): the m8 card at a real 150 MHz"). Commits are
  dated 2026-10-19 02:20-10:29 -03:00. Merge-base with the feature branch: `f237dab2`.
- `MarsWise/openTPU:main` was found at `9bd4d83c` too (a fast-forward from `5309a20d`, which is
  its ancestor; not done by an agent of this project). Feature branch on GitHub: `668dcff5`.
- Delta: 42 files, +2,954/-383. No change to `opentpu/isa.py`, `isasim.py`, `compiler.py` or
  `quant.py`, so TMM's software contract is untouched.

| Upstream area (paths) | Effect | NextGen overlap and decision | Verification |
|---|---|---|---|
| MXU (`rtl/mxu/otpu_mxu.sv`, +400): epilogue on a registered advance with a TMEM request queue (TQ), RMAX compare a stage later, `last_k` as a compare against `KBm1`, decoded block scale as a registered copy per column (`dlo0/dhi0` -> `g_bd`) | Timing at 150 MHz on m8; cycle-exact by upstream's tests | **Conflict** (2 hunks). Kept `KBm1` and our `ternary` bit in the command entry; TMM's `m0.ternary` passed to the new per-column `bdec()`; the old `always_ff` copy dropped. `KBm1` derives from `cmd_KBa`, which already counts TMM advances | TMM RTL vs ISA on the merged tree (PROGRESS.md) |
| Sequencer / pkg (`otpu_seq.sv`, `otpu_pkg.sv`): command store and VPU wait hold from flip-flops, `otpu_tmem_breq` | Timing | Auto-merged; upstream adds no opcode dispatch that bypasses `OP_TMM` (checked by diff) | same |
| TMEM / DMA / VPU / slice (`otpu_tmem.sv`, `otpu_dma.sv`, `otpu_vpu.sv`, `otpu_quant.sv`, `otpu_se_*`, `otpu_slice.sv`, `otpu_native_dram.sv`, `otpu_afifo.sv`, tech RAMs) | Reads by bank from flip-flops, LUT RAMs as read-ahead BRAMs, fanout copies | No NextGen edits in these files; taken as is | upstream tests; selected local regression |
| XDMA write packing (`rtl/shell/otpu_mem_ch.sv`) | Host-to-card write path at 150 MHz | Taken as is; affects every host write (weights, activations) | upstream `memch_test`; board only after a rebuild |
| Board build (`boards/ypcb-00338/vivado/*`, `core_clock.tcl`, `constraints/otpu_mxu_place.tcl`) | 150 MHz CLK2X clock, setup margins, TMEM/XDMA pblocks, movable MXU array, place directive | `create_project.tcl` auto-merged; our `SEQ_WC` default of 1 kept | Not rebuilt; r3 on the card predates this merge |
| Host (`opentpu/host/loadcal.py` +399, `board.py`, `ddrcal.py`) | DDR3 read taps centred under load on a configuration's first scrub when the core clock differs from the controllers' | No NextGen overlap; behaviour on the m4 133.33 MHz image to be checked at the next board bring-up | upstream test_loadcal etc. |
| ASIC flow (`asic/nangate45/*`, `tools/asic/flow.py`) | CTS / GRT repair | No overlap | not run |
| Docs (`docs/timing_ffs.md`, `docs/litedram.md`) | The timing study; r150o production at 150 MHz on m8 since 2026-10-19 | Reference | — |

Consequence for the card: the loaded r3 bitstream (`e9727303`, m4, 133.33 MHz) no longer matches the
feature sources. A rebuild (m4 at 133.33 MHz, or upstream's m8 at 150 MHz with TMM) and a new
`firmware_contract.json` are needed before the board is again "the sources' image".

## Earlier snapshot (2026-10-16)

## Snapshot and review scope

- Source repository: `FeSens/openTPU`, read-only.
- Latest fetched `main` checked 2026-10-16: `f237dab2f196855a59a083b1454786df8f01291f`,
  authored 2026-10-14 19:58:00 -03:00.
- Previous integration point: `702d6c98013f279af4b11e135c3d3e0529410021`.
- Integrated upstream delta: 100 commits, 124 files, +12,184/-2,505.
- Integration commit on the feature branch: `07eb6cd19be5521fd405ab866fc7b0683eae731e`.
- Current published feature HEAD: `668dcff5295043a2618e558fe72c44a183ebd56d`.
- Fetch verification on 2026-10-16 found no commits after `f237dab2`; the feature branch is
  15 commits ahead of that snapshot. Fork `main` remains at
  `5309a20d28948fa910897e85a96f82db6ee5c1fd`.
- GitHub API inspection found no open upstream PRs or issues at the time of this check. The
  latest-release API returned 404 (no published latest release). The upstream branch list is
  active; only the first page was inspected, not every branch.

The full, reproducible path inventory is `git diff --name-status
702d6c98013f279af4b11e135c3d3e0529410021..f237dab2f196855a59a083b1454786df8f01291f`
in the shared repository. The table lists the principal affected paths by functional group and
their compatibility disposition; the command above is the canonical exhaustive list. Paths
listed in the overlap column received additional semantic review.

## File-level disposition

| Upstream area and changed paths | Interface/architectural effect | NextGen overlap and decision | Verification |
|---|---|---|---|
| **Board / clock / DDR:** `boards/ypcb-00338/constraints/otpu_ddr_phy_place.tcl`, `boards/ypcb-00338/constraints/otpu_mxu_place.tcl`, `boards/ypcb-00338/litedram/core.json`, `boards/ypcb-00338/litedram/otpu_litedram.v`, `boards/ypcb-00338/vivado/build.tcl`, `boards/ypcb-00338/vivado/create_project.tcl`, `boards/ypcb-00338/vivado/impl_directives.tcl`, `opentpu/host/litedram/sdram_init.py`, `rtl/asic/otpu_asic_top.sv`, `rtl/dma/otpu_dma.sv`, `rtl/mem/otpu_native_dram.sv`, `rtl/mem/otpu_tmem.sv`, `rtl/shell/otpu_afifo.sv`, `rtl/shell/otpu_board.sv`, `rtl/shell/otpu_mem_ch.sv`, `rtl/tech/asic/otpu_tech_ram.sv`, `rtl/tech/generic/otpu_tech_ram.sv`, `rtl/top/otpu_slice.sv`, `rtl/vpu/otpu_se_tail.sv`, `rtl/vpu/otpu_vpu.sv`, `sim/tech/tb_tech_ram.sv`, `sim/verilator/otpu_ldc_ch.v`, `sim/verilator/otpu_ldc_mem.sv`, `sim/verilator/tb_board.sv`, `sim/verilator/tb_memch.sv`, `targets/nangate45.toml`, `tools/litedram/csr_pipe.py`, `tools/litedram/gen_core.py`, `tools/litedram/gen_ldc.py`, `tools/litedram/h2c_contention.py`, `tools/litedram/ld_test.py`, `tools/litedram/regtimer.py`, `tools/litedram/regtimer_check.py`, `tools/litedram/wl7ddrphy.py`, `tools/memch_ooc.tcl`, `tools/memch_test.py` | Board r2/r2.1/r2.2 work: placement directives, DDR PHY/LiteDRAM settings, DMA/memory arbitration and registered/fanout paths. This changes timing and memory contracts and may affect target-specific builds. | Retained from upstream. `rtl/mxu/otpu_mxu.sv`, `rtl/top/otpu_pkg.sv` and `rtl/top/otpu_slice.sv` were reviewed as TMM-adjacent. TMM dispatch and registered upstream MXU grant (`mx_gq`) are both present. No board build or claimed timing result was reproduced. | Software and selected RTL regression only; no Vivado build, synthesis, physical link or board result. |
| **ASIC implementation flow:** `asic/nangate45/fastroute.tcl`, `asic/nangate45/io_latency.tcl`, `asic/nangate45/mxu_macros.tcl`, `rtl/tech/asic/otpu_tech_ram.sv`, `rtl/tech/generic/otpu_tech_ram.sv`, `targets/nangate45.toml`, `tools/asic/fakeram.py`, `tools/asic/flow.py`, `tools/asic/paths.py`, `tools/memch_ooc.tcl` | Adds Nangate45/OpenROAD-oriented timing and memory flow; it is an ASIC estimate/flow, not a Kintex-7 utilization or timing report. | Retained. No conflicting TMM-specific resource assumption was introduced. Treat upstream reported results as upstream claims until independently reproduced. | A later selected ASIC-flow test group passed (97 pass, 2 skip) on the f237-integrated tree; no current flow run here. |
| **Compiler, kernels and models:** `opentpu/compiler.py`, `opentpu/kernels/attention.py`, `opentpu/kernels/mailbox.py`, `opentpu/kernels/mlp.py`, `opentpu/language.py`, `opentpu/llm/gemma4.py`, `opentpu/llm/lfm2.py`, `opentpu/llm/moe.py`, `opentpu/llm/mtp.py`, `opentpu/llm/prefill.py`, `opentpu/llm/qwen3.py`, `opentpu/llm/qwen35.py`, `opentpu/llm/tiled.py`, `opentpu/imgcache.py`, `opentpu/progcache.py`, `opentpu/host/board.py`, `opentpu/host/chat.py`, `opentpu/host/offload.py`, `opentpu/host/streamed.py`, `opentpu/rtlsim.py` | Adds compiler counter/derived addressing and tiled prefill support, updates Qwen/LFM/MoE/MTP/model paths, cache/offload/host execution and simulator/build integration. | `opentpu/compiler.py` and `opentpu/llm/qwen3.py` overlap NextGen TMM work. TMM compiler lowering and accounting were preserved alongside upstream counter/address changes. Shared changes passed selected compiler/ISA regressions after integration. | Source integration plus selected software tests. Framework/model numerical verification remains incomplete where PyTorch/checkpoints are absent. |
| **Serving and host I/O:** `opentpu/serve/api.py`, `opentpu/serve/kvio.py`, `opentpu/serve/scheduler.py`, `opentpu/serve/store.py`, `opentpu/host/board.py`, `opentpu/host/chat.py`, `opentpu/host/offload.py`, `opentpu/host/streamed.py` | Adds/extends serving, KV movement, batching, offload and streamed model execution. | Retained; an HTTP API is not evidence of a native vLLM platform/worker/backend. No native vLLM plugin was found. | Selected upstream tests have historical results; no current physical backend/model execution claim. |
| **RTL overlap:** `rtl/mxu/otpu_mxu.sv`, `rtl/top/otpu_pkg.sv`, `rtl/top/otpu_slice.sv`, `rtl/dma/otpu_dma.sv`, `rtl/mem/otpu_tmem.sv`, `rtl/vpu/otpu_vpu.sv`, `rtl/vpu/otpu_se_tail.sv` | Upstream changes DMA batching, TMEM/DRAM/board interfaces, VPU fanout and MXU grant timing. | The merge had conflicts in `rtl/mxu/otpu_mxu.sv`; resolution preserved local `i_ternary` reset and upstream `mx_gq` reset/logic. Shared RTL source is integrated in merge `07eb6cd1`; no blanket side preference was used. | Historical full TMM RTL suite passed at the f237-integrated tree with Verilator 5.046; this session re-ran TMM RTL tests with Verilator 5.052. |
| **Tests and qualification:** `tests/test_asic_flow.py`, `tests/test_board.py`, `tests/test_chatmoe.py`, `tests/test_compiler.py`, `tests/test_config_matrix.py`, `tests/test_em_tile.py`, `tests/test_fakeram.py`, `tests/test_host_runs.py`, `tests/test_lesim.py`, `tests/test_lfm2_moe.py`, `tests/test_minicpmv.py`, `tests/test_mtp.py`, `tests/test_mtp_moe.py`, `tests/test_perf_ddr.py`, `tests/test_prefill.py`, `tests/test_qual_refs.py`, `tests/test_qual_sh.py`, `tests/test_qwen35.py`, `tests/test_qwen35_moe.py`, `tests/test_rtl.py`, `tests/test_serve_mtp.py`, `tests/test_session_scripts.py`, `tests/test_target.py`, `tests/test_tech_ram.py`, `tests/test_tiled.py` | Adds/extends coverage for ASIC flow, board model, MTP/MoE, tiled prefill, VLM, serving, DDR and session scripts. | `tests/test_compiler.py` and `tests/test_rtl.py` overlap; conflict resolution retained both TMM tests/imports and upstream MXU/arbiter tests. | At integration time: selected compiler/ISA/ternary 75 passed; selected RTL 6 passed; selected ASIC/target/memory/qualification/script group 97 passed, 2 skipped. This does not substitute for a current full suite. |
| **Tools / datasets / docs:** `docs/asic.md`, `docs/asic_timing_loop.md`, `docs/dataflow.md`, `docs/litedram.md`, `docs/mtp.md`, `docs/mxu_systolic.md`, `docs/observability.md`, `docs/offload.md`, `docs/prefill.md`, `docs/serve.md`, `docs/tiled_prefill.md`, `tools/dn_chunk.py`, `tools/litedram/csr_pipe.py`, `tools/litedram/gen_core.py`, `tools/litedram/gen_ldc.py`, `tools/litedram/h2c_contention.py`, `tools/litedram/ld_test.py`, `tools/litedram/regtimer.py`, `tools/litedram/regtimer_check.py`, `tools/litedram/wl7ddrphy.py`, `tools/moe_prefill_cosim.py`, `tools/offload/lesim.py`, `tools/offload/mtp_bound.py`, `tools/offload/mtp_transplant.py`, `tools/offload/pack_pool.py`, `tools/offload/server_replay.py`, `tools/offload/sessions/emprol.sh`, `tools/offload/sessions/les1.sh`, `tools/offload/sessions/mtp1.sh`, `tools/offload/sessions/mtp2.sh`, `tools/offload/sessions/mtps.sh`, `tools/offload/sessions/mtpt.sh`, `tools/offload/strip_experts.py`, `tools/perf_ddr.py`, `tools/perf_prompt.py`, `tools/prefill_routes.py`, `tools/qual/chatmoe.py`, `tools/qual/qual.sh`, `tools/qual/refs.py` | Adds documentation, profiling, DDR tests, prefill routes, model/offload utilities and session qualification scripts. | Retained. These tools do not imply a verified model/checkpoint, board result or native vLLM backend. | Syntax/selected script tests were run historically; no new upstream claims adopted as local measurements. |

The path groups above overlap in a few entries (for example, LiteDRAM and qualification tools)
because they participate in more than one contract; use the Git diff command above for the
canonical unique-file inventory of 124 paths.

## Integration sequence and compatibility risks

1. `055d4f84` merged the reviewed `e15b4a4` snapshot.
2. `380ddd6e` merged the reviewed `702d6c9` snapshot.
3. `07eb6cd1` merged `f237dab2`; `fb7602d8` and `668dcff5` recorded and published status.
4. The shared TMM integration uses opcode `0x23`; existing MM numeric formats are retained.
5. `rtl/mxu/otpu_mxu.sv` has a shared grant/handshake and needs regression coverage after any
   future change; the upstream registered grant must remain cycle-consistent with TMM issue/pop.
6. `opentpu/compiler.py`, `opentpu/llm/qwen3.py`, `rtl/top/otpu_pkg.sv`, and
   `tests/test_rtl.py` are shared surfaces; future upstream merges require another semantic
   review rather than automatic file replacement.
7. No new upstream commits were found after `f237dab2` on the 2026-10-16 fetch. No new merge was
   required in this session.

## Evidence limits

The integrated generic TMM path is not validated against a specific BitNet/Tern2 checkpoint.
Upstream descriptions of board runs, timing closure, MTP, VLM and offload performance are not
measurements produced by this branch. The current read-only JTAG observation is recorded
separately in `HARDWARE_CONNECTIVITY.md`; it is not PCIe, DMA, runtime or inference evidence.

## Upstream check 2026-11-16: main `d84db557` (10 commits after `91df4109`)

Fetched 2026-11-16 (git, FeSens/openTPU main). Not merged: `git merge-tree` against the r8 branch
reports 60 conflicting files, the core RTL among them (otpu_mxu, otpu_seq, otpu_dma,
otpu_native_dram, otpu_quant, otpu_ctrl, otpu_board), while an r8 rebuild is under way.

| Upstream commit | Date | Feature | Files | Overlap with r8 | Decision | Verification |
|---|---|---|---|---|---|---|
| b2eeed69 | 2026-11-10 | Core wide-group (WG) memory path, a wider MXU option, multi-slice hubs (BCU1525) | 109 files, rtl +17.6k lines: otpu_mxu, otpu_native_dram, otpu_hbm_ch, otpu_coll, otpu_dma, otpu_quant, otpu_seq | Every r8 RTL change (TMM CQ / T5 / GRQ, OREG, QST4 / 6, k0) is in the same modules | DEFER until the r8 build with the k0 fix is accepted on the card; then port by module with RTL == ISA per step | NOT RUN |
| ff0b3e2b | 2026-11-10 | Compiler, host and server for BCU1525: grouped weights, multi-slice; moe.py, gptoss.py, batch35.py, cards.py, board_perf.py | 177 files, +31k | compiler.py, isasim.py, qwen3.py, qwen35.py, board.py, progcache.py: all changed in r8 | DEFER with b2eeed69 (its ISA contract) | NOT RUN |
| ab85880a, 44d7d4be, c2819255 | 2026-11-10 | boards/bcu1525 (VU9P, four DDR4 DIMMs), its tests and docs | boards, targets, tests, docs | none (another board); boards/bcu1525/Makefile conflicts | DEFER (no VU9P here) | NOT RUN |
| 65fc3549, a22032fc | 2026-11-10 | Caches say when they do not keep (progcache, imgcache), a card says which image it holds (cardimage.py) | host | progcache.py (r8: OTPU_PROG_CACHE_FREE_GB) | PORT LATER: small, useful on YPCB (the card naming its image) | NOT RUN |
| d67cc2ea | 2026-11-11 | docs/offload.md, mtp2.sh wording | docs | none | DEFER (docs) | - |

Checked for r8's open problems: upstream's MXU takes MM's k0 as `w4[31:24]` (the r8 erratum is
ours alone, from 6df6a665); its otpu_mem_ch change is otpu_sfifo's timing form (HCP column
groups, registered flags) for 300-450 MHz clocks, not a functional fix; boards/ypcb-00338 changes
only otpu_clk2x.tcl, the Makefile and create_project.tcl (to read when porting).

## Upstream check 2026-11-21

`git ls-remote https://github.com/FeSens/openTPU.git`: main still `d84db557` (2026-11-11); of the
192 branches the newest are `fix-labels` (11.11, merged as #11) and `bcu1525-public` (10.11,
`a22032fc`). Nothing new since the 2026-11-16 review. The merge stays deferred: our RTL is now
r5x's (the r8 timing commits taken out), so the 60 conflicts of 16.11 are to be redone against
it. The GitHub API (pull requests, issues) is not reachable from this session; ls-remote and
shallow fetches are.
