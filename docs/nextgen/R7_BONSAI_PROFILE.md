# Bonsai GSC profiling and optimization boundary

The r5 board report records one full prompt: logits cosine 0.999926239 against
the independent reference and matching next token. This is not a throughput
or long-generation qualification.

`otpu_tmm_gsc` expands a GSC operation into sequential group subcommands. The
ternary address expression reuses a packed weight chunk for four groups; the
binary expression does so for eight. Whether this dominates elapsed time or
external memory traffic must be measured. Group scales are applied before
accumulation; a PAIR/QUAD implementation must preserve each scale, accumulation
ordering and tails, stalls and completion behavior. No speedup is claimed.

`tools/bench_bonsai_gsc.py` builds the same synthetic ternary matrix with
constant per-row scales represented either as RSC or repeated GSC scales,
compares against float and ISA and can prepare board counters. Default is ISA:

```
python tools/bench_bonsai_gsc.py --backend isa --m 1 --n 128 --kb 16
```

Fourteen unit tests pass. Board mode writes DRAM and runs instructions: **do
not run it without separate user permission**. It requires explicit expected
firmware BUILD_ID and CFG_HASH, the shared card lock, initialized DDR and clean
ECC before and after each run; it does not program a bitstream. Port-B counts
are request chunks and include program/scale/other traffic, not DDR pin bytes;
subtracting a baseline does not perfectly isolate weight traffic. Uniform
synthetic scales do not represent actual Bonsai group-scale variation. Compare
multiple representative shapes and real workloads before deciding on an RTL
change; report FPGA/ISA equality, CPU model output, cycles, traffic and ECC.
