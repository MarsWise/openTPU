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

## C1: TMM K-split for single-row decode (design, 2026-10-20; not implemented)

**Bottleneck.** A TMM DRAM chunk (D bytes) carries four D-element K blocks (2 bits a weight); the
MXU consumes one block per advance per column, every column the same block of the weight row
against its own ACT row. With M = 1 (decode) three of the four columns idle: KB cycles a weight
row, 128 MAC/cycle (board r3: 127.7 MAC/cycle at M = 1, 511.6 at M >= 4).

**Cycle model, BitNet b1.58 2B4T, one token, TMM only.** 30 layers x (q 2560x2560, k/v
640x2560, o 2560x2560, gate/up 6912x2560, down 2560x6912) = 2.08 G ternary MACs:

| Mode | cycles / token | ms at 133.33 MHz | DRAM bytes / token | DRAM-bound ms at 128 B/cycle |
|---|---|---|---|---|
| today (M = 1, one column) | 16.3 M | 122 | 520 M | 30.5 |
| K-split (four columns, one chunk an advance) | 4.07 M | 30.5 | 520 M | 30.5 |

K-split makes decode DRAM-bound instead of compute-bound (4x on TMM time). For int8 MM nothing
changes: one D-byte chunk is one block, already the DRAM rate.

**Design (bit-exact: the same integer sums).**
- QACT flag KSPLIT (rows = 1): source block k goes to ACT row k mod 4, block ab + k div 4, with
  the row's one Sa and its block's da offsets (the scale records move with the block).
- TMM flag KSPLIT (M = 1): one advance per chunk; column j decodes quarter j of the chunk
  (symbols j*D .. j*D + D - 1, the existing `ts` select per column instead of per advance) against
  its ACT row's block; at the row's end the four columns' integer accumulators are added (a
  4-input 48-bit adder) and the one fp32 epilogue runs for column 0. The ISA result is TMM's
  without KSPLIT bit for bit, so isasim needs only the layout and the checks.
- RTL cost: per-column ternary decode at S1 (4 x D symbol muxes, or the raw chunk hopping 1,024
  bits a column instead of decoded 2,048), a 2:1 mux in front of each column's weight hop
  register, the QACT write addressing, the column-sum adder; WNS today is +0.011 ns, so the mux
  must sit before a register.

**Acceptance.** RTL = ISA bit-exact for TMM KSPLIT vs plain TMM on random shapes (KB not a multiple
of 4, scales, ACC, RMAX); board: TMM M = 1 at >= 480 MAC/cycle; no timing regression (WNS >= 0
at 133.33 MHz); BitNet decode TMM time per token <= 35 ms. **Precondition:** the board run of the
hybrid BitNet model must show TMM as the decode bottleneck; if the 210 host launches a token
dominate, moving a whole layer on the device (BitNet as a resident-program Spec, like the Qwen3
engine) comes first.
