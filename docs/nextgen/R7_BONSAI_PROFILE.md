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

The Hadamard activation transform is lowered to existing VPU instructions in
`opentpu/kernels/hadamard.py`; ternary group-scaled projections use TMM GSC.
In the normal chat path the host picks the first token from prefill logits.
Sampled `generate_card` currently gets uniforms from a host NumPy generator and
prepares repetition-penalty factors on the host. In addition, positions before
`_conv_lo` fall back to `Engine.step` and a host pick. These paths must move to
the device before a full FPGA-only generation claim. The new
`--strict-fpga-only` refuses the current chat rather than silently using them;
the older `--strict-fpga` has a narrower placement check.

An experimental Philox4x32-10 block is in `rtl/exp/philox`; its RTL testbench
checks Random123 known-answer vectors, response backpressure and reset. It is
not connected to the ISA, TMEM, generation loop or the r7 bitstream under
construction, and no synthesis/resource/timing result exists for it. The
runtime must manage distinct counter/key pairs, store per-request state,
convert samples to the required distribution and validate the resulting
sampling before replacing host RNG. Gaussian noise for DDPM is also not
implemented by this block.

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
