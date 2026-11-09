"""Exercise the actual MXU wsel function on converted synthetic CACT codes.

Only the function is elaborated; this is not full-MXU or board simulation.
"""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from opentpu.cactus.cq import read_cact
from opentpu.cactus.transcode import cq_to_tmm_cq
from test_cactus_cq import fixture

ROOT = Path(__file__).resolve().parents[1]


def _t5rom(src):
    """wsel's T5 table (the function and its localparam)."""
    a = src.index("  function automatic logic [2559:0] t5rom();")
    b = src.index("T5ROM = t5rom();", a) + len("T5ROM = t5rom();")
    return src[a:b]


def _function(src, marker):
    start = src.index(marker)
    end = src.index("\n  endfunction", start) + len("\n  endfunction")
    return src[start:end]


@pytest.mark.parametrize("bits,inter", [(2, False), (2, True), (4, False), (4, True)])
def test_actual_mxu_wsel_matches_cact_codebook(tmp_path, bits, inter):
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        pytest.skip("Icarus Verilog and vvp are required")
    path = tmp_path / "synthetic.cact"
    fixture(path, bits, inter)
    folded, packed, _, lut = cq_to_tmm_cq(read_cact(path))
    book = np.pad(lut, (0, 16 - len(lut))).astype(np.int8)
    vectors = []
    for row in range(packed.shape[0]):
        for block in range(2):
            expected = lut[folded.indices[row, 128*block:128*(block+1)]]
            vectors.append(f"{block:02x}{int.from_bytes(book.tobytes(),'little'):032x}"
                           f"{int.from_bytes(packed[row,:128].tobytes(),'little'):0256x}"
                           f"{int.from_bytes(expected.tobytes(),'little'):0256x}\n")
    vec = tmp_path / "vectors.hex"
    vec.write_text("".join(vectors), encoding="ascii")
    source = (ROOT / "rtl/mxu/otpu_mxu.sv").read_text(encoding="utf-8")
    dec4 = _function(source, "  function automatic logic [7:0] dec4(")
    wsel = _function(source, "  function automatic logic [7:0] wsel(")
    cqa = _function(source, "  function automatic logic [3:0] cqa(")
    t5 = _t5rom(source)
    tb = tmp_path / "tb.sv"
    tb.write_text(f'''`timescale 1ns/1ps
module tb;
 localparam int D=128;
 localparam logic [1:0] WF_W8=2'd0;
 {t5}
 {dec4}
 {wsel}
 {cqa}
 reg [2183:0] vec[0:{len(vectors)-1}]; integer v,i; reg [7:0] got;
 initial begin
  $readmemh("{vec.as_posix()}",vec);
  for(v=0;v<{len(vectors)};v=v+1) begin
   for(i=0;i<D;i=i+1) begin
    got=wsel(vec[v][2047:1024],i,1'b0,2'd0,1'b1,1'b0,vec[v][2178:2176],
             1'b1,1'b{1 if bits == 4 else 0},
             vec[v][2048 + 8*cqa(vec[v][2047:1024],i,1'b0,1'b{1 if bits == 4 else 0},
                                 vec[v][2178:2176]) +: 8],1'b0);
    if(got!==vec[v][8*i +: 8]) $fatal(1,"case %0d lane %0d",v,i);
   end
  end
  $display("PASS: actual MXU CQ{bits} decoder"); $finish;
 end
endmodule
''', encoding="ascii")
    exe = tmp_path / "test.vvp"
    result = subprocess.run([iverilog, "-g2012", "-Wall", "-s", "tb", "-o", str(exe), str(tb)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"iverilog exit {result.returncode}: {result.stdout}{result.stderr}"
    result = subprocess.run([vvp, str(exe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"vvp exit {result.returncode}: {result.stdout}{result.stderr}"
    assert f"PASS: actual MXU CQ{bits} decoder" in result.stdout
