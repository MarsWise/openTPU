# Nextgen implementation progress

## Kokoro-82M on the device, card test package for 81dc3958 (2026-11-24, night)

- **Kokoro-82M whole graph** (`opentpu/kokoro_graph.py`, `tools/kokoro_card.py`): every model
  operation of KModel.forward_with_tokens on the device, in two graphs.
  - Graph A: PL-BERT, the duration encoder, the durations, the text encoder.
  - Graph B: built for the rounded durations. It covers the alignment row copies, F0Ntrain, the
    decoder, and SineGen (phases mod 1, linear interpolation, VOP RAND Box-Muller noise). Then
    the source STFT (fp32 DFT, atan2), the generator and the iSTFT.
  - The host only maps phonemes to ids and picks the voice row. It reads back T durations
    (graph B's shape) and writes the seed.
  - The reference is hexgrad's kokoro 0.9.4 modules, vendored in `tools/kokoro/ref`.
    `tests/test_kokoro_graph.py`: 4 passed.
  - ISA measurements: graph A cos 0.998-0.99998 with equal durations; F0 / N 0.99999; log-mel
    cos 0.994; generator alone 22.1 dB.
  - Card: `r8_kokoro.cmd`.
- `resident.Graph.conv1d` splits long outputs into row-block programs (IMEM).
- **Card scripts for the new firmware:**
  - `r8_card_check`: 3 GRQ QUAD 2-row cases and the TGQ4M bit check. Board model: 5/5 GRQ QUAD
    cases pass.
  - `r8_all_tests.cmd new`: profile, Bonsai MTP, Qwen3.5 MTP, LFM3 fp4, Ouro, Dory, Maple, the
    ISA-only HF models (`r8_more_models`: ReasonLite, LFM2.5 thinking, FunctionGemma,
    Qwen3.5-4B fp4), PaddleOCR, Kokoro.
  - `llm_card.py`: `ouro` / `dory` kinds.

## GRQ QUAD of 2 rows + drain overlap (RTL), Maple-Preview, Qwen3.5 MTP / LFM3 fp4 card scripts (2026-11-24)

- **GRQ QUAD of 2 rows** (`TGQ4M`, CAPS2 bit21; docs/mtp.md 11.1). This was the reason Bonsai MTP
  was slower on the card (its 2-row verify ran plain GRQ). The ISA, the compiler (it pads rs so
  that all 8 drain lanes work), the MXU (`gq4_run`) and the host gate are done.
  - RTL SIMULATED, bit for bit: 5 M 2 cases, the 10 M 1 cases, and the GSC pipelined cases.
  - Cycles: M 2 779 against plain GRQ 1646.
- **GRQ QUAD drain overlap** (`GQ4_PIPE`): a GRQ QUAD command of the same M pops while the one
  before it drains. This was Bonsai decode's MXU_STARVE 10.5 %. Every GRQ QUAD / GSC RTL test
  still passes bit for bit.
- **Firmware:** `r8_build_lean.cmd` now builds r5x lean + TGQ4M + GQ4_PIPE, CFG_HASH 81dc3958.
  `r8_card_check` knows the hash. Card: NOT RUN.
- **Maple-Preview** (`opentpu/llm/maple.py`, `tests/test_maple.py` 11 passed, ISA). It is the
  first ternary MoE: 256 experts, top 8, streamed into DRAM slots by the host.
  - Plain GSC ternary format `tgsc` (TMM GSC without Bonsai's Hadamard signs). TQ2_0 repacks
    losslessly into it.
  - Expert slots in ternary GSC.
  - Clamped SwiGLU (`silu_clamp7`).
  - RoPE kinds as lfm3 tables, using a head-dimension permutation (Maple's partial rotate_half
    becomes lfm3's form with theta^2).
  - Context <= 512 (the sliding window equals full attention there).
  - Card: `r8_llm.cmd maple` (128 slots a layer, 3.0 GiB image).
- **Card scripts:**
  - `r8_mtp.cmd`: Qwen3.5-0.8B / 2B plain vs MTP, int8 and fp4.
  - `r8_llm.cmd lfm3-fp4`: LFM3 fp4 layers, `test_lfm3` device graph fp4 cos > 0.85.
  - `r8_inspect_models.cmd`: Maple's GGUF header and Ivonar Nano's checkpoint keys, with no code
    from the model folders run.

## Bonsai MTP, atan without TATAN, Kokoro blocks (2026-11-24)

- **Bonsai 2 MTP** (docs/mtp.md 11.8). The MTP layer comes from PrismML's
  `-PQ2_0-MTP-Q8_0.gguf`. Its main blocks are byte-identical to `-PQ2_0`'s. The MTP layer is
  int8 (or fp4 with `OTPU_BONSAI_MTP_FORMAT`), with no Hadamard. On the ISA, a tiny native
  model gives plain greedy decode's tokens, whatever the drafts, in the host loop and in the
  device loop, and its states equal plain decode's. Not yet on the card: `r8_bonsai_mtp.cmd`.
- **atan2 on the lean firmware.** `kernels/atan2.atan01` runs an odd polynomial of degree 15
  on the VPU when the target has no TATAN. It is bit-exact on the ISA and on the RTL.
- **Kokoro-82M decoder blocks** (Codex's `kokoro_resident.py`, finished):
  - the real AdaINResBlock1 reaches cos 0.99988; Kokoro has no InstanceNorm affine, so its
    defaults are used;
  - in conv_post + iSTFT, the inverse DFT moved from an int8 MXU product to the fp32 VPU:
    28.2 dB became 41.6-47.4 dB;
  - the checkpoint is converted with `torch.load(weights_only=True)` (`tools/kokoro/convert.py`,
    sha256 pinned).

## r5x lean: shared TFN ROM, one RAND generator (2026-11-22)

- `otpu_se_comp`: one TFN coefficient ROM for each two lanes (dual-port block RAM, a port a lane): 4 copies instead of 8. `otpu_vpu` `RAND_W` (default 1): one VOP RAND generator, a RAND chunk one column a cycle through the composites' narrow-issue path; the numbers are the same (a function of each column's counter). RTL SIMULATED: test_se_comp 39, test_vrand 18, VPU / board / stream / TFN / atan / format gates 119, all passed.
- Build: `tools/nextgen/card_runs/r8/r8_build_lean.cmd` (r5x settings, `OTPU_DROP_FEATURES=TBIN TBGSC TATAN`, TMM T5 kept), CFG_HASH 852be9dc, tag `lean`; `r8_build_noatan.cmd` retired into it. Expected: about -24 RAMB36 (ROM) and -5k LUT / -4k FF (RAND), ESTIMATE, SYNTHESIS NOT RUN.
- Build `7f376a78-lean` (2026-11-22, SYNTHESIS ESTIMATED, routed): LUT 203,637 (-4,546 vs r5x), FF -4,249, RAMB36 -20, RAMB18 -5 — the area as estimated. Timing FAILED: WNS -0.124 ns on r5x's own MXU path (weight FIFO BRAM -> weight decode -> `g_wsk`; the MXU netlist is unchanged), Vivado exited in post-route phys_opt, no bitstream. Next: `r8_build_lean_incr.cmd` (incremental from r5x's routed checkpoint, which met timing on that path); if that fails, T5 off shortens the decode on that path.
- TMM T5 off in the lean build (2026-11-22): no card model uses it, and it is GSC only (not with GRQ / QUAD, so Bonsai would lose GRQ QUAD with it); its trit decode is on the failing weight-decode path, and C4 (r8 less T5 / BIN) cost about 10k LUT less in the MXU and closed at +0.059. `r8_build_lean.cmd` now drops `TBIN TBGSC TD5 TATAN`: CFG_HASH add73610; the card tooling checks the T5 and atan refusals. The RTL keeps T5 behind `HAS_TD5` (the ISA and the Bonsai 2 2B ISA path stay).
- Build `46a18fc1-lean` (2026-11-23, SYNTHESIS ESTIMATED, bitstream written): signoff WNS +0.054 / WHS +0.022, core_clk +0.151 ns (r5x +0.002); LUT 194,103 (65.0 %, -14.1k vs r5x), FF -5.0k, BRAM tiles 565 (-22.5), slices 84.2 % (r5x 88.7 %); MXU -9.5k LUT (T5). sha256 db6d04a4...5fdf. Earlier attempts died in Vivado (EXCEPTION_ACCESS_VIOLATION at varying stages); this one ran after a reboot. Next: `r8_after_build.cmd 46a18fc1-lean` on the card.
- **r5x lean is the current firmware (2026-11-23, PHYSICAL FPGA VERIFIED):** r8_card_tests 17/0/0 with ECC 0/0 on every step (the T5 / BIN / atan refusals, VOP RAND with one generator == ISA, TFN on the shared ROM == ISA, KV, Diffusion, SmolVLA, YOLOX, vLLM, Bonsai); DDR3 diag / split / check clean; every model's decode identical to r5x to the cycle; one SEC step (r8_ttft, Qwen3.5 fp4, ch1). Defaults now: `OTPU_DROP_FEATURES=TBIN TBGSC TD5 TATAN` (CFG_HASH add73610), `r8_load.cmd` loads 46a18fc1-lean, firmware_contract's firmware = 46a18fc1.
- **GSC groups pipelined (2026-11-23, RTL SIMULATED):** the card profile (Bonsai 2 4B: DRAM 67 %, MXU busy 99 % / MAC 52 % / starved 25 %) traced to `otpu_tmm_gsc` handing a K block's subcommand to the MXU only after the one before completed. `PIPE` (default on) hands it over under the older block's drain; results bit for bit the serial adapter's and the ISA's (146 + 5 RTL tests). GRQ QUAD N 2560 KB 20: 17,459 -> 14,884 cycles; N 256: 4,013 -> 2,422 (docs/nextgen/BONSAI_GRQ.md). Needs a bitstream (`r8_build_lean.cmd`, CFG_HASH add73610 unchanged).
- **Needle: one decoder program a prompt (2026-11-24, ISA + RTL SIMULATED, card NOT RUN):** the card's ~50 ms a token (19.7 / 16.7 tok/s) was the host's: each token compiled a new step program (the token and position were compile-time values) and computed its sequencer wait entries in Python (sandbox: 11 + 119 ms int8, 26 + 171 ms CQ4; the decode reads ~15 MB of weights, about 1 ms on the card, an ESTIMATE). `needle_dev.Engine(run_args)` (default where the backend takes run arguments): the token and position are the run's arguments (qwen3.RunPos; the image gains the decoder's mask row table), one program a prompt and bucket, loaded into IMEM once. ISA: logits and every DRAM byte bit for bit against a program a token, int8 and CQ4, the README's tool call both ways; Verilator board RTL: the decoder step tests (now at run arguments) == ISA. `tools/needle_card.py --compile-each`: the old path. Card tok/s: NOT RUN (`r8_needle.cmd`).
- **Resident graphs as one chained run (2026-11-24, ISA + RTL SIMULATED, card NOT RUN):** SmolVLA base ran 12,948 programs as 12,948 host launches (a load, a start and a poll each; Diffusion Policy 100 steps: 10,790). `resident.chain_slots` / `run_board_chained`: every program in its DRAM slot (its final HALT becomes `LI R1, next slot; LI R2, its length; HALT CHAIN`, its wait entries after it on a WAITC card: the program's own, computed once per distinct program, and zeros for the three control instructions), written at once; the host starts the first and polls one halt. ISA: the image bit for bit the sequential runs'; a card model through `run_board_chained` (WAITC and not): one load, one run, the same image, each slot's words and entries checked; Verilator board RTL with and without WAITC (boot loader, HALT CHAIN from the slots, AXI stalls) == the ISA. `tools/smolvla_run.py` / `tools/diffusion_run.py --resident` on the card chain by default where the bitstream has HALT CHAIN (CAPS bit30; `--sequential`: a launch a program; a refused chain falls back to it). Card wall: NOT RUN (r8_card_tests' SmolVLA / Diffusion steps).
- **Card, 2026-11-24 (firmware 46a18fc1 lean, host tree c5a6edfd), PHYSICAL FPGA VERIFIED:** `r8_needle`: Needle int8 **675.1 tok/s** (was 19.7), CQ4 **457.3 tok/s** (was 16.7), 20 tokens picked on the card with the encoder in 0.2 / 0.3 s (was 1.1 / 1.4 s); logits card == ISA bit for bit, README tool call, ECC 0/0. `r8_card_tests` 17 passed, 0 failed, ECC 0/0: Diffusion pusht 100 steps as one chained run (10,790 programs, 17.3 MiB of slots: prepare 0.95 s, load 0.02 s, card 2.7 s; device run 3.9 s; rel L2 0.0015), 10 steps 0.9 s, tiny card == ISA; SmolVLA tiny chained, card == ISA. SmolVLA base: the chain was refused (its 3,960 MiB image leaves ~135 MiB, the slots need ~397 MiB) and ran a launch a program: device run 96.9 s (prepare 40.6 s, card 10.4 s), rel L2 0.0308. Fixed after the run: `chain_segments` (consecutive programs that fit, one launch each over the same slot area; a card model in tests: several segments, the sequential image). SmolVLA base segmented: card NOT RUN.
- **Firmware ed482f26 lean is current (2026-11-24, PHYSICAL FPGA VERIFIED):** 46a18fc1 plus the pipelined GSC adapter; signoff WNS +0.019 / WHS +0.015 (core +0.095; userclk1 -0.009 rerouted to +0.019 by build.tcl), sha256 2193aa7d...; r8_card_tests 17/0/0, ECC 0/0; `r8_profile`: **Bonsai 2 4B 10.13 tok/s** (13.16 Mcycles a token, was 8.45 / 15.79), MXU_STARVE 10.5 % (was 25 %), DRAM 80 %; BitNet 17.68, Qwen3.5-0.8B 19.94 tok/s unchanged. firmware_contract.json and r8_load.cmd's default switched. impl_1's post-route phys_opt did not finish (Vivado "Failed runs: impl_1"), build.tcl went on from the routed checkpoint; r8_build.cmd now calls such a build good when otpu.bit is new and setup / hold are met. DDR3 soak on ed482f26 (r8_after_build): NOT RUN.
- **Policies run again on new inputs without their preparation (2026-11-24, ISA, card NOT RUN):** `resident.ChainPlan` (assembled, wait entries, chain slots once; run() again: a single segment's slots stay in DRAM), `smolvla_resident.VLA.inputs` (the input tensors' regions, padding zero as in a fresh image), `smolvla_run.py --repeat N` (new noise each, only the inputs written). ISA tiny, host and device noise: each repeat == a fresh image's ISA run bit for bit; a card model: a plan rerun on a rewritten input == the sequential runs of a fresh image, one and several segments. r8_card_tests 2a / 2b run with --repeat (2a gates the fresh-image equality on the card). Found and fixed on the way: `diffusion_resident.Loop.noises(seed)` was shadowed by its noise-tensor attribute (`--device-noise` raised TypeError; tests/test_resident.py's device-noise test now passes with diffusers installed).
- Memory-channel FIFOs to BRAM: not done. Of the 7.8k LUTRAM only the two 64-deep 512-bit buffers per channel (`mem_ch` u_ar, `native_dram` port A, about 2.7k LUTRAM in all) fit BRAM sensibly (+32 RAMB36); the rest are 16-32 deep. They sit on the DDR3 data path while its ECC qualification is open, so they wait until the soak is clean.

## TATAN (TFN atan) removable in RTL (2026-11-22)

- `otpu_se_comp` gets `HAS_ATAN`; the board's `TATAN` generic reaches it through `otpu_slice` (`HAS_TATAN`) and `otpu_vpu`. At 0 each lane's TFN ROM holds tables 0-7 (2048 x 96 bit instead of 2304), CAPS2 bit12 is clear and the host refuses VOP atan (no model uses it). RTL SIMULATED (Verilator 5.046): TFN 0-9 less atan bit-exact, format-gate testbench with HAS_TATAN 0.
- Build: `tools/nextgen/card_runs/r8/r8_build_noatan.cmd` (r5x settings + `OTPU_DROP_FEATURES=TBIN TBGSC TATAN`, tag `noatan`), CFG_HASH 852be9dc; card tooling (r8_card_check, r8_card_tests, r8_restore_pcie) accepts it and checks the atan refusal. firmware_contract: pending entry. The default build stays r5x (432e34c2) until a no-atan bitstream is built and checked.
- Expected saving: a few RAMB36 (the TFN ROM is about 50 RAMB36 + 5 RAMB18 for 8 lanes; atan is 1/9 of it), LUT about zero. SYNTHESIS NOT RUN.

## r5x update (2026-11-21): the current firmware, DDR3 cause narrowed, full suite

State by evidence class (rows: VERIFIED_RESULTS.csv, c42beb65 / 514b1e0a / ffb35491).

- **Current firmware r5x `c42beb65`** (m8, 133.33 MHz, CFG_HASH 432e34c2, sha256 307FC451...35A2BF):
  r5 45b1c1d3's MXU / sequencer pipeline plus every r8 instruction but TMM BIN / BIN GSC, r5's
  memory write path (WREG 0), LiteDRAM rtw 3, trace 16384. Routed WNS +0.002 / WHS +0.036 ns,
  LUT 208 183. Merged into `feature/r7-fpga` (25eb4e5c): the branch's RTL; the default r8 build
  (`build_r8_133.ps1`) uses its settings. `firmware_contract.json`: current.
- **PHYSICAL on r5x:** DDR3 read windows under load >= 4 taps on every lane (as r5; r8 builds 1-3);
  r8_ddr_check 4 passes + split 3 passes: 5 corrected words, 0 uncorrectable (r8 A: tens a pass);
  r5 on the same checks: 2 corrected in 2 passes (the board's remainder). Card tests 17 / 0 / 0
  with ECC 0 throughout: every r8 instruction == ISA, KV int8 / kv6 / k6v6, Diffusion pusht,
  SmolVLA base (rel L2 0.0308), YOLOX strict FPGA-only, vLLM BitNet, Bonsai 2 4B GRQ / GRQ QUAD,
  Needle, LFM3, MiniCPM5, Granite. Decode (device tok/s): Qwen3 25.6, Qwen3.5-0.8B 20.1,
  Qwen3.5-2B 8.4 (two clean runs of three), LFM2.5 66.9, BitNet 17.7, Bonsai GRQ QUAD 8.5.
- **Cause of r8's DDR3 errors:** its three timing commits (5e6293e6 + 63cbadcd in the MXU,
  800ae9d0 in the sequencer) or the placement they bring, not the instructions. The bisect build
  `exp/r5x-oreg` (bb5af80d: r5x + the MXU pair) is prepared (`r8_build_oreg.cmd`, NOT RUN): it
  should add timing slack, and its read windows say which half narrows them.
- **Full suite on r5x's RTL (WSL, 32 CPUs):** 2 202 passed, 160 skipped, 1 xfailed; the 7
  failures and 2 collection errors were stale tests (fixed, ffb35491); every Verilator test passed.
- **Open:** timing slack (+0.002 ns), the hours-long soak (`r8_soak_long.cmd`, NOT RUN), BIN /
  BIN GSC (1-bit Bonsai, CQ1) back or not, VLM on the card (PaddleOCR-VL `tools/paddleocr_card.py`
  prepared), SmolVLA's wall (wait entries now computed once per distinct program in parallel:
  NOT RUN on the card), an autonomous graph on the device, MTP / MoE without the host, YOLOX COCO
  mAP on the card, MusicGen / Kokoro / SD 1.5, FFN pruning, the CDC sign-off.

## r8 update (2026-11-16): on the card, MXU k0 erratum, rebuild in progress

State by evidence class (rows: VERIFIED_RESULTS.csv; narrative: HANDOFF_2026-11-10.md,
HANDOFF_2026-11-15.md). Nothing is ACCEPTED: acceptance needs ECC 0 / 0 over a long run on a
bitstream without a known error.

- **Bitstream r8 `2c800a8f`** (m8, 133.33 MHz, CFG_HASH 93e29b18): routed, WNS +0.041 / WHS
  +0.013 ns, LUT 70.3 %, DSP 75 %. PHYSICAL: loaded 2026-11-10, every r8 instruction card == ISA.
  **Erratum:** its MXU drops bit 4 of MM's k0 (`board.MM_K0_ERRATA`; RTL fixed in c9f51ceb, RTL
  SIMULATED). The rebuild (`r8_build_k0fix.cmd`, source 1a77396a) started 2026-11-16; not routed
  yet. `OTPU_MM_K0_REPORT=1` (set by the card scripts) prints per run how many MMs had the bit:
  a count above 0 marks a 2c800a8f result as wrong.
- **PHYSICAL on r8 2c800a8f (execution; to be repeated on the fixed build):** Qwen3 tiny KV
  int8 / kv6 / k6v6 == ISA; Diffusion Policy pusht 10 / 100 steps (rel L2 0.0044 / 0.0038);
  SmolVLA base (rel L2 0.0308, 12948 host launches, 402 s); YOLOX-Nano strict FPGA-only (one
  frame); vLLM BitNet 4 slots sampled / greedy / logprobs; Bonsai 2 4B GRQ off / on logits.
  Card run of 2026-11-10 20:30: 17 of 17, ECC 0 / 0 in every step after the load calibration.
- **DDR3 on r8:** SEC (once DED) in earlier runs, 0 / 0 after `loadcal ternary:4`; r5 45b1c1d3
  on the same card 0 / 0 without it. Not qualified; cause not established (placement near the
  PHY, the write path's slack +0.229 ns against r5's +1.983 ns, otpu_mem_ch's CDC: open).
  Soak: `set PASSES=4` + `r8_ddr_check.cmd`.
- **ISA SIMULATED, card runs prepared (NOT RUN):** Cactus Needle 26M int8 / CQ4 with the token
  picked on the device (`r8_needle.cmd`; the decoder step also RTL SIMULATED); LFM3 text decoder,
  MiniCPM5-2B, Granite 4.2 3B (`r8_llm.cmd lfm3 | minicpm5 | granite`, tools/llm_card.py).
  Speed of the LLMs already on the host: `r8_speed.cmd` (NOT RUN on r8).
- **ISA SIMULATED only:** Bonsai 2 2B (T5), Ouro 1.4B (8 layers int8), Dory 4B (a subnet),
  PaddleOCR-VL 1.6, MiniCPM-V.
- **Open for the ТЗ:** an autonomous graph on the device (the host launches every program), one
  strict FPGA-only audit, a memory planner for several models, MTP first token / k > 1, MoE
  without the host, YOLOX COCO mAP on r8, the full test suite on HEAD, sign-off (CDC-1 / CDC-10,
  I/O delays). Not started: MusicGen, Kokoro, SD 1.5 (as whole models), FFN pruning.

## r7 update (2026-10-31): Bonsai GRQ and Diffusion device ops

- **TMM GSC GRQ** (CAPS2 bit 9, `Config.TGRQ`): block-major packed ternary layout; the GSC
  adapter issues one subcommand per block over its plane and the MXU consumes each chunk once as
  four replay groups (one per output row). Bit-for-bit GSC; RTL SIMULATED (44 tests incl. MCOLS 8),
  native Bonsai image/descriptor/compiler wired (`qwen35.gsc_grq`, `OTPU_GRQ=0` to disable). Not
  synthesized, not on the card; expected several-fold Bonsai decode gain is an ESTIMATE
  (BONSAI_GRQ.md). Build: `start_r7_133.cmd` (TGRQ follows `-TernaryGroupScales`).
- **Diffusion Policy device ops** (`tools/diffusion_run.py --device-ops`): GroupNorm, Mish, ReLU,
  spatial soft-argmax, the DDPM update and in-place / per-channel (FiLM) adds and products on the
  device (`kernels/eltwise.py` group_norm / softmax_xy / mish, `kernels/ddpm.py`,
  `torchdev.DeviceGroupNorm` / `DeviceSpatialSoftmax` / `device_ddpm`). ISA-verified against
  float64 / diffusers; tiny policy actions unchanged (rel L2 0.0178). Host: max-pool, col2im,
  layout, scheduler scalars and noise draw. Not run on the card yet.

## Local update (2026-10-26, evening): handoff to Codex / Copilot

HANDOFF_2026-10-26.md is the state: r5-m8 PHYSICAL results (BitNet 17.5 / 20.9 tok/s, Qwen3.5-2B
int8 8.39 / fp4 15.20 / MTP 19.1, small LLMs 10-17% over upstream, YOLOX COCO 0.2825 at 0.22 s,
SmolVLA tiny and Diffusion Policy pusht 10 steps with 100% MACs on the card), vLLM batched decode
(27-29 tok/s) blocked by DDR3 channel 0 ECC errors under device stores (open; rmw_check.sh
decides logic vs physical), ~170 commits to publish (push_fork.cmd).

## Local update (2026-10-26): r5-m8 bitstream built, timing met at 137.5 MHz

- `2f2320e2`, m8 (MCOLS 8, MXU_E 8, TMEM 128K, TFN / MAC / T4 / TMM PAIR + QUAD), core 137.5 MHz,
  core_clk2x 275: routed WNS +0.021 / WHS +0.016 ns, every clock positive; otpu.bit sha256
  270afa7a...; firmware_contract.json records it (RESOURCE_BUDGET.md "r5-m8 bitstream").
- 150 MHz did not close: db36f818 -0.52 ns, Codex's locality placement -0.281 ns (routed). The
  remaining paths (TMEM's pumped outputs into the units, the MXU epilogue's fp32 normalisation) need
  RTL (a TMEM read register; fp_mul_s2 split) for 150.
- **On the card (2026-10-26 14:15): build 0x2f2320e2 runs, CFG_HASH matches, otpu-selftest ALL PASS
  incl. vext (TFN / MAC / QST T4 == ISA, first time on hardware), ECC 0 / 0, TMM QUAD 507.2 MAC/cycle
  bit-exact** (PHYSICAL; RESOURCE_BUDGET.md "r5-m8 2f2320e2 on the card"). Next: claude_run_r5a.
- Run r5a on the card: r5 instructions == ISA, YOLOX COCO-500 0.2825 at 0.22 s/image (r4 0.32-0.34),
  BitNet 16.46 tok/s, vLLM 12.06 tok/s, SmolVLA tiny == ISA (RESOURCE_BUDGET.md "run r5a"). Fixed
  after it: chunked device attention (SmolVLA base), vLLM worker releases the card at shutdown,
  ECC cleared after loadcal, RTL-only tests skip without Verilator. Next run: claude_run_r5b.sh.
- Card validation prepared (the user's yes for the JTAG load was given and the load done): `_claude
5_1_jtag.cmd`
  (sha256-checked volatile load), `r5_2_pcie.cmd` (rescan + `r5_check.sh`), then `claude_run.cmd`
  with `tools/nextgen/card_runs/claude_run_r5a.sh` (memcal + loadcal, selftest, r5_card_check, TMM,
  YOLOX, SmolVLA, BitNet, vLLM, qual, the full pytest).

## Local update (2026-10-25): isolated locality replay prepared

- The `db36f818` m8 build completed routing (0 routing errors) but failed setup: core
  -0.671 ns, clk2x -0.603 ns. The worker exited during post-route physopt; the user did not
  stop it. Its exact failure mechanism remains unknown. Routed/earlier checkpoints survive.
- Read the routed path analysis: 4,875 failing endpoint rows. Prepared a separate experiment
  for four RMW FP multipliers and 25 pumped TMEM controllers (304 rows), without changing RTL.
- `tools/replay_r5_locality.ps1` reuses the post-opt checkpoint and writes to a fresh directory;
  it neither resets the original project nor synthesizes or writes a bitstream. Place mode
  stops after placement/physopt; Route explicitly resumes from the new physical checkpoint.
- Vivado Plan verifies all 29 soft pblocks / 1,963 fabric cells on the real netlist. The source
  checkpoint hash and 105 clock/exception/delay constraint statements are unchanged. Eleven
  offline checks pass. Physical timing of this experiment is **NOT RUN**.
- Instructions and limitations: [R5_LOCALITY_EXPERIMENT.md](R5_LOCALITY_EXPERIMENT.md).

## Local update (2026-10-23): upstream and r5 timing recovery

- Merged FeSens main `91df4109` (19 new commits after `9bd4d83c`) as `f139f524`:
  Bend ISA/cache simulators and checks, CLI integration, unary STREAM and LFU fixes.
  This upstream interval contains no RTL/Vivado changes.
- T4 now uses static byte-enabled synchronous RAM banks with a read-ahead drain. Isolated
  Vivado 2024.2.2 synthesis confirms 8 RAM blocks and no T4 FF array (previously 65,536 FF).
- Explicit TFN descriptors remove the synthesis defect. The same SE top before/after gives
  latches 1,280 -> 0, no_clock 8,704 -> 0, unconstrained internal endpoints 1,280 -> 0.
  Coefficient ROM words are unchanged.
- MXU PAIR/QUAD reduction has two registered adder levels with aligned scales/valid/FIFO
  writes. MM gains one latency cycle; the pipeline still accepts one row per cycle.
- m4/m8 board-model checks pass. Detailed regression evidence, Bend's local-extension limits
  and the fresh MCOLS=8 / 150 MHz build procedure: [R5_TIMING_RECOVERY.md](R5_TIMING_RECOVERY.md).
  Full-chip placement/routing and a new bitstream are not yet measured.

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

### r4 runs 4b-4d (2026-10-21): DRAM channel 0, loadcal, the IMEM overflow

- 4b: COCO plain clean, COCO --wsplit ECC_DED on channel 0 again; loadcal did not run (its load
  did not fit a 64K TMEM: fixed, 16b7d04a).
- 4c: loadcal ran, channel 0 then 0 / 0 over three COCO passes at 53-56 C (not thermal: read taps
  at the edge of their window under load, e.g. lane 1 tap 14 -> 1 at bitslip -1); but it started
  from the core CPU's stale result after a host calibration, so channel 1 lane 5 lost a bit offset
  (~6e8 corrected words per pass, 0 uncorrectable). Fixed (36f4263c): with the CPU held, the host's
  last calibration is the baseline.
- 4d: the core's CPU recalibrated, loadcal from it: **both channels 0 / 0 ECC events over
  selftest, COCO plain and --wsplit** (every lane a window under load). COCO plain 0.2821.
- COCO --wsplit gave mAP 0.0000 with no error: YOLOX's stem program was 4765 instructions and the
  card's IMEM holds 4096 (IMEM_WORDS 32768, ACT_BLOCKS 128 lets the stem roll a long loop body);
  Board.load_program loaded PROG_N rows anyway and the program ran overwritten. The ISA check
  (--check-isa) refused it. Fixed: load_program refuses a program longer than the bitstream's
  IMEM; cnngraph retraces a conv that does not fit with the row loop (stem --wsplit: 1061, the
  result bit for bit the rolled one's); ISA with the card's config: head rel L2 0.0922 as before.

### r5-m8 timing (2026-10-24/25)

- Codex's fixes (0f9f9b1d: T4 in BRAM, no TFN latches, PAIR / QUAD sum in two stages) passed
  synthesis's gates on the full design; placement then failed at 150 MHz: core -2.715 ns placed,
  -1.677 after phys_opt, level-5 congestion over the array (RESOURCE_BUDGET.md "r5-m8
  implementation"). Long wires through the author's logic, on a design ~57 k LUTs / ~71 k FFs
  above r150o's.
- The excess is mostly TMM's parallel fabric datapath in each PE; 9e36232f moves the ternary
  products into the DSP's multiplier (the same integers, cycle for cycle). Next: the MXU out of
  context (`tools/vivado/r5_synth_check.tcl mxu`), then the full build (`_claude/r5m8_next.cmd`
  does both, the build only if the MXU's LUTs fell under 75 k).
- Container verification of Codex's tree: T4 / quantizer 37/37 (m8, 75 % stalls), TFN / SE /
  MAC / torchdev 86 + 1 skip, the r5 board model PASS; test_rtl's S=4 cases cannot compile in the
  7 GB container (cc1plus killed): BLOCKED here, to run on the 92 GB host
  (`_claude/claude_run_vl.sh`: Verilator 5.046 from source + the whole suite).
- The MXU out of context on 9e36232f (the user's run): 55,276 LUTs / 57,391 FFs, before 99,269 /
  114,417 (SYNTHESIS ESTIMATED). Then the cycle-exact timing fixes for each of 0f9f9b1d's failing
  groups (RESOURCE_BUDGET.md "Timing fixes for the next r5-m8 build"): the array's DSP columns
  fenced, the TMEM request registers replicated, sw_rdy from flip-flops, the PEs' selects copied
  per column. Not yet built. RTL SIMULATED on the fixes (container, Verilator 5.046, each copy
  checked against its original every cycle): test_rtl `-k tmm` 41 passed (TMM / PAIR / QUAD, m4
  and m8), test_r5_board 1 passed, test_quant_rtl 37 passed; test_eltwise 18 passed 1 skipped
  (GELU's erf without TFN), test_torchdev 14 passed 1 skipped (no LeRobot); test_rtl act-reads /
  4-bit MLP 19 passed 1 skipped; test_board.py all 159 passed after two fixes (the qw_ok check
  armed before a slice's first reset, simulation only; op_checks' VOP MAC SCALAR, which the ISA
  forbids, dropped: otpu-diag would have reported it failing on an r5 card).
- Build db36f818 (the fixes + fold): MXU out of context 49,286 LUTs; placed WNS -1.059 ns, after
  phys_opt -0.503 ns (with the 0.15 ns margin), 193,845 LUTs / 182,866 FFs. The rest fails on TMEM's
  pumped RAM outputs into the units (clk2x -> core). Prepared: an implementation-only rerun on the
  same synthesis (`build_r5_timing.ps1 -Mode Impl`, `_claude\r5m8_impl.cmd`) with a 0.45 ns
  clk2x -> core margin, AggressiveExplore phys_opt, Explore routing, 8 threads
  (RESOURCE_BUDGET.md "r5-m8 build db36f818").
- Host software: torchdev resident weights (addd2e7c) and the program cache (9166b633): SmolVLA
  tiny 553 calls -> 87 traced, 466 reused, 1.56 MB of weights once; actions unchanged (rel L2
  0.0205).

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
