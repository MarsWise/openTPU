"""INT6 packed group expansion to signed W8 bytes; standalone RTL only."""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from opentpu.nextgen.kv6 import pack_rows

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/exp/kv_int6"


def test_all_codes_at_all_positions_and_random_groups(tmp_path):
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        pytest.skip("Icarus Verilog and vvp are required")
    rows = []
    for lane in range(16):
        for value in range(-32, 32):
            row = np.zeros(16, dtype=np.int8)
            row[lane] = value
            rows.append(row)
    rows.extend(np.random.default_rng(0x16).integers(-32, 32, (1024, 16), dtype=np.int8))
    rows = np.stack(rows)
    packed = pack_rows(rows)
    vectors = tmp_path / "vectors.hex"
    vectors.write_text("".join(f"{int.from_bytes(r.tobytes(),'little'):032x}"
                               f"{int.from_bytes(p.tobytes(),'little'):024x}\n"
                               for r, p in zip(rows, packed)), encoding="ascii")
    tb = tmp_path / "tb.sv"
    tb.write_text(f'''`timescale 1ns/1ps
module tb;
 reg [95:0] payload; wire [127:0] values; reg [223:0] vectors[0:{len(rows)-1}]; integer i;
 otpu_kv_int6_unpack16 dut(.packed_i(payload),.values_o(values));
 initial begin
  $readmemh("{vectors.as_posix()}", vectors);
  for (i=0; i<{len(rows)}; i=i+1) begin
   payload=vectors[i][95:0]; #1;
   if(values!==vectors[i][223:96]) $fatal(1,"group %0d got %h expected %h",i,values,vectors[i][223:96]);
  end
  $display("PASS: {len(rows)} groups"); $finish;
 end
endmodule
''', encoding="ascii")
    exe = tmp_path / "test.vvp"
    result = subprocess.run([iverilog, "-g2012", "-Wall", "-s", "tb", "-o", str(exe),
                             str(RTL / "otpu_kv_int6_codec.sv"),
                             str(RTL / "otpu_kv_int6_unpack16.sv"), str(tb)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "implicit definition of wire" not in result.stderr, result.stderr
    result = subprocess.run([vvp, str(exe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PASS: {len(rows)} groups" in result.stdout
