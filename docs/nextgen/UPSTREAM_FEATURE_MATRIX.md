# Upstream feature and compatibility matrix

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
