"""Test proposed INT6-to-W8 mapping in RTL decode + ISA, not MXU RTL."""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from opentpu import fp32 as F, isa as I
from opentpu.isasim import Config, Machine
from opentpu.nextgen.kv6 import pack_rows, quantize_rows

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/exp/kv_int6"


@pytest.mark.parametrize("kb", [1, 3])
def test_decoded_codes_and_shared_scale_through_w8_rsc_isa(tmp_path, kb):
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        pytest.skip("Icarus Verilog and vvp are required")
    d, m, n = 128, 2, 4
    rng = np.random.default_rng(2200 + kb)
    weights = rng.normal(size=(n, kb * d)).astype(np.float32)
    weights[0] = 0
    weights[-1, -1] = -30  # one scale for the entire row, across every block
    q, scales = quantize_rows(weights)
    packed = pack_rows(q).reshape(-1, d * 3 // 4)
    vectors = tmp_path / "groups.hex"
    vectors.write_text("".join(f"{int.from_bytes(p.tobytes(),'little'):0{d*3//2}x}\n" for p in packed), encoding="ascii")
    tb = tmp_path / "tb.sv"
    tb.write_text(f'''`timescale 1ns/1ps
module tb;
 reg [{d*6-1}:0] payload; wire [{d*8-1}:0] values; reg [{d*6-1}:0] groups[0:{len(packed)-1}]; integer i;
 otpu_kv_int6_unpack_block #(.D({d})) dut(.packed_i(payload),.values_o(values));
 initial begin
  $readmemh("{vectors.as_posix()}",groups);
  for(i=0;i<{len(packed)};i=i+1) begin payload=groups[i]; #1; $display("ROW %h",values); end
  $finish;
 end
endmodule
''', encoding="ascii")
    exe = tmp_path / "decode.vvp"
    result = subprocess.run([iverilog, "-g2012", "-Wall", "-s", "tb", "-o", str(exe),
                             str(RTL / "otpu_kv_int6_codec.sv"),
                             str(RTL / "otpu_kv_int6_unpack16.sv"),
                             str(RTL / "otpu_kv_int6_unpack_block.sv"), str(tb)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"iverilog exit {result.returncode}: {result.stdout}{result.stderr}"
    result = subprocess.run([vvp, str(exe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"vvp exit {result.returncode}: {result.stdout}{result.stderr}"
    output_rows = [line[4:] for line in result.stdout.splitlines() if line.startswith("ROW ")]
    assert len(output_rows) == len(packed), result.stdout
    decoded = np.frombuffer(b"".join(int(line, 16).to_bytes(d, "little")
                                      for line in output_rows), dtype=np.int8).reshape(n, kb * d)
    assert np.array_equal(decoded, q)

    x = rng.normal(size=(m, kb * d)).astype(np.float32)
    cfg = Config(S=1, D=d, MCOLS=4, ACT_BLOCKS=8, DRAM_BYTES=1 << 20)
    wa, sa, out = 0x20000, 0x30000, 4096
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    dram[:x.nbytes] = x.view(np.uint8).reshape(-1)
    dram[wa:wa + decoded.size] = decoded.view(np.uint8).reshape(-1)
    dram[sa:sa + scales.nbytes] = scales.view(np.uint8).reshape(-1)
    prog = [I.ld(0, 0, x.size), I.qact(0, m, 0, kb, kb * d),
            I.mm(wa, sa, out, n, kb, kb * d, n, m, 0, 4, wf=I.W8, rsc=True), I.halt()]
    sl = Machine(cfg, [prog], [dram]).run().slices[0]
    got = sl.tget(out + np.arange(m * n)).reshape(m, n)

    # Independent scalar group accumulation using ACT's observed quantized values.
    # This checks W8/RSC scale mapping, including activation group shifts and
    # the simulator's FP32 rounding order; it is not a floating-model accuracy test.
    expected = np.zeros((m, n), np.float32)
    for row in range(m):
        for col in range(n):
            accum = 0
            for block in range(kb):
                for group in range(4):
                    shift = 16 - int(sl.adoff[row, block, group])
                    a = sl.act[row, block*d + group*(d//4):block*d + (group+1)*(d//4)].astype(np.int64)
                    w = q[col, block*d + group*(d//4):block*d + (group+1)*(d//4)].astype(np.int64)
                    if shift >= 0:
                        accum += int(a @ w) << shift
            v = F.mul(F.i2f(np.array(accum, dtype=np.int64)), scales[col])
            expected[row, col] = F.mul(v, F.mul(sl.ascale[row, 0], np.float32(2.0 ** -16)))
    assert np.array_equal(got.view(np.uint32), expected.view(np.uint32))
