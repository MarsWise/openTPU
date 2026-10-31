# Bonsai 2 4B: measured r5 baseline, 2026-10-31

Physical YPCB-00338, firmware build `45b1c1d3`, CFG_HASH `a5bcbaef`,
133333 kHz, D=128, MCOLS=8, two DDR3 channels. Model
`prism-ml/Ternary-Bonsai-2-4B-mlx-2bit`, ternary GSC for layers and head,
cap 512, greedy, prompt `Write a short story.`, seed 0. The exact checkpoint,
tokenizer and prompt hashes and returned token IDs are in
`evidence/bonsai-r5-20261031/bonsai-r5-before.json`.

The prompt has 17 tokens. The host selected the first reply token; the device
then generated 31 further tokens in one run. ECC SEC/DED was 0/0 on both
channels at preflight, before decode and after the run. These boundary samples
do not establish long-run memory reliability or prove no internal ECC clear.

| Metric | Measurement | Boundary |
|---|---:|---|
| Decode wall rate | 0.6805 token/s | 31 device-generated tokens, includes start/compilation overhead after first host pick |
| Decode rate from FPGA cycles | 0.7061 token/s | 188.833 million RUNNING cycles/token |
| Steady arrival rate | 0.7061 token/s | First to last device-generated token |
| Prefill wall | 27.21 s | 17 prompt tokens, excludes model loading |
| Full measured turn | 72.76 s | Includes prefill and decode, excludes model load |
| Controller read events | 17.896 GB/token | 64-byte events across both channels, all traffic in decode interval; not an isolated weight or GSC count |
| Port-B requests | 33.35 million/token | All B requests during the captured run, not exclusively GSC |
| MXU_BUSY / RUNNING | 99.76% | MXU has an active instruction; not MAC utilization |
| MXU_MAC / RUNNING | 17.41% | Cycles consuming chunks; not a count of useful model operations |
| MXU_STARVE / RUNNING | 17.52% | FIFO empty only; systolic partial-row waiting is not included |

Host-and-other wall minus device RUNNING time is 1.65 s over the 45.55 s decode;
that subtraction includes program setup, waits and overlapping work, not only
CPU arithmetic. First device-token start took 3.07 s, including 1.65 s in
program compilation. The sustained device rate and wall rate are close. This
is one run and is not a long-duration stability or independent logits test.

## Isolating packed GSC rereads

Separate controlled microbenchmarks used identical synthetic ternary weights
and a single scale per row duplicated across all groups. Both modes matched
ISA bit-for-bit and ECC was 0/0. The `isolated` columns subtract a separate
QACT-only baseline, so they are approximate; they do not measure DDR pin bytes.

| N | KB | Mode | Port-B requests after baseline | Cycles after baseline |
|---:|---:|---|---:|---:|
| 128 | 16 | RSC | 512 | 2,098 |
| 128 | 16 | GSC | 2,048 | 5,483 |
| 2,560 | 20 | RSC | 12,800 | 51,247 |
| 2,560 | 20 | GSC | 51,200 | 349,185 |

The 2,560×2,560 shape is representative of the model's hidden width but the
data are synthetic. Separate real Bonsai groups have different FP32 scales:
the RSC case is a diagnostic, not a valid replacement for their math. The
observed fourfold port-B requests agree with the serial adapter's packed weight
access pattern. The controller-wide counter and MXU activity support investigating
memory and serial issue, but do not establish which single change will improve
end-to-end tokens/s. Do not infer the measured microbenchmark ratio as a model
speedup.

## Implementation constraints

The GSC adapter currently executes all N rows of group k, then group k+1. It
cannot retain every packed row in the small FIFO for arbitrary N. A tiled reuse
experiment needs to account for output addresses, ACC and RMAX, command ordering,
stalls, reset, overlapping DMA, and independent FP32 scales and rounding. Existing
PAIR/QUAD add integer partial results before scaling, so they cannot simply be
enabled for different GSC scales. Preserve the r5 bitstream and compare any
candidate on the same model, tokens, clock and configuration after ISA/RTL tests,
synthesis, routing and separate permission to program the FPGA.

An experimental instruction tiler (`opentpu/nextgen/gsc_tiling.py`) splits an
unpaired ternary GSC operation into output-column tiles, preserving the full K
order for every output and the original ACC behavior. Seventeen ISA tests cover
varying scales, tails, M and ACC; two RTL tests compare the tiled sequence
against the original ISA with AXI stalls and WAITC off/on. It refuses RMAX and
incompatible flags. It is not connected to the compiler and does not yet retain
or reuse any weight: tiling alone does not lower the request count or establish
a speedup. A cache/replay must also preserve command and memory ordering.

The r7 source `09e9d34f` synthesis passed, including structural T4/latch gates,
but its implementation crashed inside Vivado phys_opt and produced no bitstream.
A separate diagnostic placement replay is in progress; it is not an optimized
firmware or a timing result.
