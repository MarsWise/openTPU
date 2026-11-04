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

## C1: TMM K-split for single-row decode (design 2026-10-20; step 1, TMM PAIR = K-split by two, and step 2, TMM QUAD = K-split by four, implemented: see "C1 step 1" / "C1 step 2" below)

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

**RTL measurement (2026-10-20, Verilator 5.046, `tools/tmm_cycles.py`, D=128 MCOLS=4, TMM of N
rows x K = 2560 against M ACT rows, cycles of the TMM alone):**

| M | N = 64 | N = 128 | MAC/cycle |
|---|---|---|---|
| 1 | 1,321 | 2,601 | 124-126 |
| 2 | 1,322 | 2,602 | 248-252 |
| 4 | 1,324 | 2,604 | 495-503 |

TMM time does not depend on M: KB = 20 cycles per weight row (one K block an advance) plus
~40 cycles of fill — the r3 card's 127.7 MAC/cycle at M = 1 is the same number. The weight
stream at M = 1 is 640 B per 20 cycles = 32 B/cycle, a quarter of the DRAM path's 128 B/cycle:
K-split would take it to 128 B/cycle. Evidence class: RTL SIMULATED.

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


## C1 step 1 (2026-10-20): TMM PAIR, the K-split by two — RTL SIMULATED

The MXU already had MM PAIR for 4-bit weights: a chunk's two blocks in one advance, columns
`j < M` the even block from ACT row `j`, columns `j + M` the odd one from row `j + M` (QACT DUP),
the partners' integers added before the result FIFO. Its PEs take two weight streams (lo / hi)
and a per-column select, and their ternary path (add / sub / skip) already honours the select.
TMM PAIR reuses all of it: a ternary chunk (four blocks) is consumed in two advances of two
blocks instead of four advances of one. No new datapath, no new register stage:

| Change | Where |
|---|---|
| TMM accepts flags bit 6 (PAIR) with `2M <= MCOLS` | `rtl/mxu/otpu_mxu.sv` (command checks, `cmd_pair`) |
| advances a row `ceil(KB/2)`; chunk index `ck >> 1`, chunk done every 2nd advance; chunk requests every 2nd advance | `ckc`, `cdone`, `c_rch`, `adv_b`, the issuer's `chunk_addr` step |
| the low stream decodes quarter `2*(ck mod 2)`, the high stream the next one | `m0.ts`, the `wd` decode |
| reserved-symbol assertion covers both quarters (not past an odd KB) | simulation check |
| CAPS2 bit1 TPAIR (a build parameter, reported) | `rtl/shell/otpu_ctrl.sv`, `otpu_board.sv`, `opentpu/host/regs.py`, `board.device_config` |
| `Config.TPAIR` (in CFG_HASH only when on: earlier bitstreams keep their hashes) | `opentpu/isasim.py`, `opentpu/target.py`, `targets/*.toml` (false: production bitstreams predate it) |
| ISA: `I.tmm(pair=)`, isasim TMM PAIR (rows j / j + M by block parity) | `opentpu/isa.py`, `opentpu/isasim.py`, `docs/isa.md` |
| compiler: a ternary dot of `M <= MCOLS/2` rows on DUP'd ACT rows emits TMM PAIR when `cfg.TPAIR` | `opentpu/compiler.py` |

**Measured (Verilator 5.046, `tools/tmm_cycles.py`, D = 128, MCOLS = 4, N = 128, K = 2560, cycles
of the TMM alone):**

| M | plain TMM | TMM PAIR | MAC/cycle plain -> PAIR |
|---|---|---|---|
| 1 | 2,601 | **1,321** | 126.0 -> 248.1 |
| 2 | 2,602 | **1,322** | 251.9 -> 495.7 |
| 4 | 2,604 | (2M > MCOLS: n/a) | 503.3 |

Correctness: `tests/test_rtl.py::test_tmm_pair_rtl` (M 1 / 2, KB 1, 4, 5, 7, 20, UNIT and RSC,
ACC, RMAX, AXI port stalls 25%): RTL = ISA bit-exact, DRAM and TMEM; `tests/test_ternary.py`: the
ISA's TMM PAIR = plain TMM bit for bit, the compiler's lowering by `Config.TPAIR`;
`tests/test_bitnet_engine.py::test_tmm_pair_engine_is_bit_identical`: the BitNet engine's decode
programs are all TMM PAIR with TPAIR and its logits equal plain TMM's bit for bit.

Effect (cycle model of section C1 with the measured rate): BitNet b1.58 2B4T decode TMM time
122 -> ~61 ms/token at 133.33 MHz (the DRAM bound stays 30.5 ms). Step 2 (K-split by four: four
weight streams through the array, 4x) needs a wider hop register or per-column decode: not done.

Evidence class: **RTL SIMULATED**. Not synthesized: the change adds no register stage or wide
datapath (one more 2:1 select of the decoded quarter, TMM terms in the chunk-count / done /
request logic), but timing (WNS +0.011 ns on r3) and resources are unverified until a Vivado
build; production bitstreams report CAPS2 bit1 = 0 and the host never emits TMM PAIR to them.

**Building a TPAIR bitstream** (not run: needs Vivado on the Windows host, separately
authorized): `TPAIR=1 boards/ypcb-00338/run_vivado.sh ...` (or `TPAIR = true` in the target's
[arch]) sets otpu_board's `TPAIR` generic, so CAPS2 bit1 reads 1 and CFG_HASH includes TPAIR
(ypcb-00338 m4 r3 knobs: 4ffd1e17 without, 39fc485e with); the host's `device_config` then turns
on `Config.TPAIR` and the compiler emits TMM PAIR. Acceptance on the card: TMM at M = 1 >= 240
MAC/cycle (`tools/tmm_cycles.py`'s workload as a board run), WNS >= 0 at 133.33 MHz, BitNet
engine logits identical to a TPAIR = 0 run.


## C1 step 2 (2026-10-20): TMM QUAD, the K-split by four — RTL SIMULATED

The design above (one advance a chunk, the four quarters in the four columns) with the layout
done by the quantizer instead of a QACT re-layout: **QACT DUP4** writes the operand's rows also to
ACT rows `M..4M-1` (the ACT RAM's write select takes two more copies, no extra cycles), and
**TMM QUAD** (TMM flags bit 4, the reserved weight-format field) has column `j + q*M` read block
`4c + q` from row `j + q*M`. The PEs get the chunk's raw codes instead of a decoded weight:

| Change | Where |
|---|---|
| TMM QUAD: `4M <= MCOLS`, not with PAIR; one advance a chunk (`ceil(KB/4)` a row), a chunk request and FIFO pop every advance | `rtl/mxu/otpu_mxu.sv` (`cmd_quad`, `cmd_KBa`, `ckc`, `cdone`, `c_rch`, `adv_b`, the issuer) |
| the low weight stream carries position i's four 2-bit codes `{q3,q2,q1,q0}` (8 bits: the stream is not widened) | `wd` at S1 |
| each PE picks quarter `{sel2, sel}` (per command, per column: `j / M`) and decodes it on the existing ternary path (fabric add / sub / skip) | `otpu_pe` in `rtl/tech/{generic,xilinx7,xilinxus}` (`sel2`, `quad`) |
| per-column ACT block: `r_blk .. r_blk4` by a 2-bit select (replaces PAIR's `r_hi`); quarters past `KB` re-read the advance's first block, their products masked | `otpu_actram`, `act_q0..3`, `m0.ov` |
| row `j < M` adds partners `j + M`, `j + 2M`, `j + 3M` (a 4-input 48-bit add before the result FIFO) | `rowi` |
| the pop head moves to the next command only if QUAD and M match (the per-command selects are read late) | `pn` |
| QACT DUP4 (flags bit 4 with DUP) | `otpu_quant` `act_dup4`, `otpu_actram` `sel()` |
| CAPS2 bit2 TQUAD, `Config.TQUAD` (in CFG_HASH only when on), targets `TQUAD = false`, `run_vivado.sh TQUAD=` | `otpu_ctrl`, `otpu_board`, `regs`, `board`, `target`, `create_project.tcl` |
| ISA `I.tmm(quad=)`, `I.qact(dup4=)`, isasim; compiler: a stationary of `M <= MCOLS/4` rows is DUP4'd and its ternary dots emit TMM QUAD when `cfg.TQUAD` (else PAIR / plain as before) | `opentpu/isa.py`, `isasim.py`, `compiler.py`, `docs/isa.md` |

**Measured (Verilator 5.046, `tools/tmm_cycles.py`, D = 128, MCOLS = 4, N = 128, K = 2560, cycles of
the TMM alone):**

| M | plain | PAIR | **QUAD** | MAC/cycle |
|---|---|---|---|---|
| 1 | 2,601 | 1,321 | **681** | 126.0 -> 248.1 -> **481.2** |
| 2 | 2,602 | 1,322 | (4M > MCOLS) | 251.9 -> 495.7 |
| 4 | 2,604 | — | — | 503.3 |

681 = 128 rows x 5 chunks + ~41 fill: one chunk (D bytes = 128 B, the DRAM path's width) a cycle,
the weight stream now at the DRAM rate. The C1 acceptance (TMM at M = 1 >= 480 MAC/cycle) holds
in RTL simulation; on the card it needs a TQUAD bitstream.

Correctness: `tests/test_rtl.py::test_tmm_quad_rtl` (KB 1, 3, 4, 5, 6, 7, 20; UNIT and RSC; ACC;
RMAX; a plain TMM and a QUAD back to back; AXI port stalls 25%): RTL = ISA bit-exact, DRAM and
TMEM, 7 passed; `tests/test_ternary.py`: ISA QUAD = PAIR = plain bit for bit (MCOLS 8, M 1 / 2,
KB 1..10), the lowering by `TQUAD` / rows; `tests/test_bitnet_engine.py::
test_tmm_quad_engine_is_bit_identical`: every decode projection a TMM QUAD, logits identical.

Effect (cycle model, BitNet b1.58 2B4T, one token at 133.33 MHz): TMM 122 -> ~31 ms (the DRAM
bound, 30.5 ms); with the int8 LM head (128,256 x 2,560: ~19 ms, DRAM-bound) and the rest (~3 ms)
a step of ~52 ms instead of the measured 145.8 ms: ~19 tokens/s of card time. The LM head is then
the next item (a 4-bit head with MM PAIR: ~10 ms; accuracy to be measured).

Evidence class: **RTL SIMULATED**. Timing risk for the Vivado build: the 4-input 48-bit partner
sum is one cycle (PAIR's was 2-input) and the PE's ternary select gains a 4:1 code mux; WNS on r3
is +0.011 ns. If the build misses, the sum takes a register (LROW + 1). Resources: ~MCOLS x D
two-bit selects and registers in the PEs, three 16-bit block addresses to the ACT RAM.

## R-series (2026-10-20): RTL acceleration for r5, ranked by the measured profile

Profile (RTL SIMULATED, Verilator, board uarch m4 D=128 MCOLS=4 MXU_E=4, `opentpu/profile.py`,
script `wk/prof.py` in the session scratchpad; tree `444c32ad`):

| kernel | MAC/cycle | busy (MXU / QUANT / VPU / DMA) | the time goes to |
|---|---|---|---|
| conv3x3 C128 16x16 (SiLU) | 276.5 | 0.72 / 0.31 / 0.39 / 0.16 | MM 163k, QACT 42k, VOP.fill 18k, exp2 13k, recip 9k cycles |
| linear 16x1024x1024 | 406.4 | 0.95 / 0.05 / 0.09 / 0.12 | MXU-bound (fill 16k: the bias) |
| sdpa H2 L128 d64 | 34.8 | 0.65 / 0.74 / 0.28 / 0.16 | QST 71k (V^T's strided bytes, one a cycle), exp2sub 32k |
| silu 16x1024 (5 VOPs) | 0.79 elem/cycle | VPU 0.80 | exp2 6.2k, recip 4.1k, mul 4.1k, add 2.1k |

| id | block | status | why |
|---|---|---|---|
| R2 | TFN: table functions VOP 0x20..0x29 (sigmoid, SiLU, tanh, GELU erf/tanh, softplus, Mish, QuickGELU, sin, cos) on otpu_se_comp, BRAM tables | RTL SIMULATED (this commit) | every family's activations: 1 VOP at 4 col/cycle instead of 5-8 VOPs at ~1 |
| R2b | conv2d: two output tiles (VPU/store of one overlaps the MXU of the other) | SOFTWARE (this commit) | the VPU's bias fill + activation stop serializing with the MMs |
| R6 | QST transposed writes (V^T: four rows' bytes a word a cycle) | PLANNED | attention prefill everywhere (ViT/VLM/SD/VLA/LLM prefill): QST 59% of sdpa |
| R3 | fused row/column affine VOP (LayerNorm, RMSNorm, GroupNorm, AdaLN) | PLANNED | 6 passes -> 3 for LN |
| R4 | MAC VOP (dst += A * C(c)) / depthwise engine | PLANNED | depthwise 2 passes a tap -> 1 (YOLO-nano/v10/v11, MobileNet) |
| R7 | m8 (8 MX columns) at 133 MHz | PLANNED | 2x MACs; upstream r2.1 does not close timing yet |

The fused bias + activation in the MXU epilogue (R1 in the first list) is deferred: with TFN at
4 columns a cycle and double-buffered output tiles the activation overlaps the MXU (the conv's
MXU produces ~0.25 outputs a cycle), so the epilogue fusion would save little against its cost
(a pipelined TFN per epilogue lane and an MM encoding change: MM's flags are fully allocated).
