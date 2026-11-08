"""Experimental INT6 expansion at existing MXU D-block widths."""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from opentpu.nextgen.kv6 import pack_rows

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/exp/kv_int6"


@pytest.mark.parametrize("d", [16, 32, 128])
def test_block_width_and_lane_order(tmp_path, d):
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        pytest.skip("Icarus Verilog and vvp are required")
    rng = np.random.default_rng(8000 + d)
    rows = rng.integers(-32, 32, size=(256, d), dtype=np.int8)
    # Exercise every lane, including the first/last across every 24-bit boundary.
    rows[0] = np.arange(d, dtype=np.int16).astype(np.int8) % 64 - 32
    rows[1] = -32
    rows[2] = 31
    packed = pack_rows(rows)
    vectors = tmp_path / "vectors.hex"
    vectors.write_text("".join(f"{int.from_bytes(r.tobytes(),'little'):0{d*2}x}"
                               f"{int.from_bytes(p.tobytes(),'little'):0{d*3//2}x}\n"
                               for r, p in zip(rows, packed)), encoding="ascii")
    tb = tmp_path / "tb.sv"
    tb.write_text(f'''`timescale 1ns/1ps
module tb;
 reg [{d*6-1}:0] payload; wire [{d*8-1}:0] values;
 reg [{d*14-1}:0] vectors[0:255]; integer i;
 otpu_kv_int6_unpack_block #(.D({d})) dut(.packed_i(payload),.values_o(values));
 initial begin
  $readmemh("{vectors.as_posix()}",vectors);
  for(i=0;i<256;i=i+1) begin
   payload=vectors[i][{d*6-1}:0]; #1;
   if(values!==vectors[i][{d*14-1}:{d*6}]) $fatal(1,"bad block %0d",i);
  end
  $display("PASS D={d}"); $finish;
 end
endmodule
''', encoding="ascii")
    exe = tmp_path / "decode.vvp"
    result = subprocess.run([iverilog, "-g2012", "-Wall", "-s", "tb", "-o", str(exe),
                             str(RTL / "otpu_kv_int6_codec.sv"),
                             str(RTL / "otpu_kv_int6_unpack16.sv"),
                             str(RTL / "otpu_kv_int6_unpack_block.sv"), str(tb)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "implicit definition of wire" not in result.stderr, result.stderr
    result = subprocess.run([vvp, str(exe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PASS D={d}" in result.stdout


@pytest.mark.parametrize("d", [1, 17, 31])
def test_invalid_width_fails_elaboration(tmp_path, d):
    iverilog = shutil.which("iverilog")
    if not iverilog:
        pytest.skip("Icarus Verilog is required")
    result = subprocess.run([iverilog, "-g2012", "-s", "otpu_kv_int6_unpack_block",
                             f"-Potpu_kv_int6_unpack_block.D={d}", "-o", str(tmp_path / "invalid.vvp"),
                             str(RTL / "otpu_kv_int6_codec.sv"), str(RTL / "otpu_kv_int6_unpack16.sv"),
                             str(RTL / "otpu_kv_int6_unpack_block.sv")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "otpu_invalid_int6_block_width" in result.stdout + result.stderr
