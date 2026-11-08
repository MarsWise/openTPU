# Experimental signed INT6 four-value codec

This standalone combinational codec is **not** part of an openTPU firmware,
ISA, checkpoint format, or KV cache. It performs no quantization, scaling,
rounding, saturation, masking, streaming, memory access, or arbitration.

## Proposed local bit contract

Four signed 8-bit inputs must lie in `[-32, 31]`. Their six-bit two's-complement
codes `c0..c3` occupy bits `[5:0]`, `[11:6]`, `[17:12]`, `[23:18]` of a
24-bit word. `byte0` contains bits `[7:0]`, followed by `byte1` and `byte2`.
Thus `[0, -1, -32, 31]` encodes as the bytes `c0 0f 7e` in address order.

If any input is out of range, pack returns `valid=0` and three zero bytes; the
caller must use `valid` and handle the invalid input. All 24-bit words decode
to four signed 8-bit values via sign extension. Unknown/X inputs are outside
this binary contract; the codec does not certify them as valid values.

There are no clocks or resets. The data are combinational and are not
registered; an integration must define latency, valid/ready, stalls, memory
byte enables, signed scale format, token/lane order, and partial groups.
These choices are not inferred from this experiment. FIFO and AXI interfaces
must not be assumed from these ports.

## Icarus verification

From the project root in PowerShell, create `build` first if necessary:

```powershell
& 'C:\iverilog\bin\iverilog.exe' -g2012 -Wall -s tb_kv_int6_codec `
  -o build\kv_int6_test.vvp rtl\exp\kv_int6\tb_kv_int6_codec.sv `
  rtl\exp\kv_int6\otpu_kv_int6_codec.sv
& 'C:\iverilog\bin\vvp.exe' build\kv_int6_test.vvp
```

The default test covers all 16,777,216 packed words against an independent
integer sign-and-shift reference, repacks each result, then checks all 256
8-bit patterns independently on each of the four input lanes. It also checks
the example above. The output must end in `PASS`; a partial run is not an
exhaustive pass. For a fast smoke, add
`'-Ptb_kv_int6_codec.WORD_LIMIT=4096'` to the compiler arguments. That smoke
is not exhaustive.

The initial Icarus full run completed successfully: 16,777,216 packed words and
1,024 input range cases. Its output is in `build/logs/kv_int6_full_initial.log`
and its compiled simulation is `build/kv_int6_test.vvp`. That run used the
initial testbench, before adding the `WORD_LIMIT` option and fixed example;
the codec RTL was unchanged at that time. The current testbench's fixed example passed in
the separate 4096-word smoke (`build/logs/kv_int6_smoke.log`).

After those tests, input ports were declared `wire` for Vivado compatibility.
The updated source passed both the 4096-word smoke and a new exhaustive run:
16,777,216 packed words and 1,024 range cases, with exit code 0. See
`build/logs/kv_int6_full_wire.log` and `build/logs/kv_int6_wire_smoke.log`.
Mandatory RTL review/signoff and full-design synthesis are separate requirements.
Icarus test success alone does not demonstrate fit, timing, numerical quality,
device integration, or hardware correctness.

## Experimental byte-write stream

`otpu_kv_int6_write_stream.sv` buffers one group and sends three 32-bit
word-addressed, byte-strobed writes. A source transfer happens on
`in_valid_i && in_ready_o`; each byte is accepted on
`out_valid_o && out_ready_i`. Output remains stable under backpressure. Input
ready does not depend on output ready. A group with an invalid signed value or
start address above `0xfffffffd` produces one cycle of `reject_o` and no write.

This is deliberately a slow functional prototype, not QST6 or an AXI master.
The output `ready` indicates **acceptance, not memory acknowledgement**; there
is no group-complete or ack port. `reject_o` has no ready signal, so the caller
must always capture it. Three byte writes are not atomic. Reset aborts a pending
group and can leave previously accepted bytes committed. Integration must define
quiescence, outstanding writes, atomicity, restart, and synchronous reset.

Current directed Icarus run: 266 groups, varying data/alignment/stalls, competing
input during stalls, invalid value and address, and reset; a byte-enable memory
model checks preservation of adjacent bytes across a 64-byte boundary: PASS
(`build/logs/kv_int6_stream_test.log`). Compile and run from the repository root:

```powershell
& 'C:\iverilog\bin\iverilog.exe' -g2012 -Wall -s tb_kv_int6_write_stream `
  -o build\kv_int6_stream_test.vvp rtl\exp\kv_int6\tb_kv_int6_write_stream.sv `
  rtl\exp\kv_int6\otpu_kv_int6_codec.sv rtl\exp\kv_int6\otpu_kv_int6_write_stream.sv
& 'C:\iverilog\bin\vvp.exe' build\kv_int6_stream_test.vvp
```

See `STREAM_REVIEW_STATUS.md`. The absent required lint hook means manual review
is required; the stream must not be reported as signed off or integrated.

Vivado 2024.2 out-of-context post-synthesis for the isolated stream and its
pack submodule completed with 73 LUT, 60 registers, no BRAM and no DSP. Reports
are in `build/logs/kv_int6_stream_ooc_*.rpt`. This is not a full r8 build or
place-and-route: 67 inputs and 69 outputs lack I/O delays, and the DRC reports
CFGBVS-1. Resource and timing results cannot be treated as device signoff.

An additional experimental 16-value word formatter emits three full 32-bit
word writes with aligned base address; see `WORD_STREAM_STATUS.md`. It is
independent of production QST and has no memory completion or atomicity.
The separate untagged acknowledgement wrapper described in
`ACKED_STREAM_STATUS.md` tracks three requests/completions, but requires a
verified real-memory contract, dependency protection and drained reset.
The isolated final RNE rounding stage and its numerical limitations are in
`QST6_ROUND_STATUS.md`; row-scale computation is not implemented.
