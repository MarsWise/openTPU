# Heterogeneous core design-space exploration

## Run status

Updated 2026-10-19: the ternary row below reflects the TMM implementation and board measurements.
The rest of this section is the earlier (2026-10-14) record.

No candidate RTL cores were added and no DSE sweep was run in this task. No Vivado or board access
was attempted. Thus the table below records decisions and evidence gaps, not synthesized estimates.

| Workload | Current bottleneck | Candidate alternatives | Traffic/cycles/resource/timing evidence | Decision |
|---|---|---|---|---|
| Existing LLM decode | README reports historical decode as DRAM-bound on listed runs; not re-profiled here | Existing MXU / dataflow tuning / more matrix columns / dedicated decode unit | No current trace, cycles, post-route result, or contention measurement in this task | Do not add a core without a fresh baseline trace |
| Existing LLM prefill | README reports matrix-rate benefits for prior configurations; not re-profiled here | Existing MXU / wider tiles / buffer and schedule changes | No workload-specific current data | Retain baseline pending controlled comparison |
| Ternary GEMV/GEMM | Decode (M = 1) uses one of four MXU columns: 127.7 MAC/cycle vs 511.6 at M >= 4 (board, r3) | Shared MXU (implemented: TMM) / N-split of one row across columns / dedicated add-sub lanes | Board r3 (133.33 MHz): 136.4 GOPS at M = 16, 34.1 GOPS at M = 1, bit-exact with ISA. The PE multiplier remains (no DSP saving claimed). | TMM reuses the shared MXU (decided, implemented). Next DSE item: M = 1 decode utilisation, with a cycle model before RTL |
| YOLO/CNN | No graph/checkpoint | Implicit GEMM / line-buffer stream core / existing MXU compositions | No model operators or trace | No decision |
| Attention/KV | No new trace | Existing stream/vector path / fused attention core | No current per-operator trace or physical resource margin | No decision |

## Reproducibility contract for future runs

Every DSE record must include commit SHA, board/part/profile, model/checkpoint hash, exact command,
tool versions, random seed, input shapes, RTL simulation result, workload trace, bytes/cycles,
resource estimate or synthesis report, timing status, and the add-vs-share acceptance decision.
Keep modeled, RTL-simulated, synthesis-estimated and user-measured physical results in separate
columns. Never populate physical results from source estimates.
