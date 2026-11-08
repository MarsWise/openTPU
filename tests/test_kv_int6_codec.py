"""Icarus smoke for the experimental int6 codec; not an integrated QST/MXU test."""
from pathlib import Path
import shutil
import subprocess
import sys
import importlib.util

import numpy as np
import pytest
from opentpu.nextgen.kv6 import pack_rows, quantize_rows


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl" / "exp" / "kv_int6"


def test_qst6_scale_round_pipeline_icarus(tmp_path):
    """Check pipelined multiply/round vs independent float32 and integer oracle."""
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        pytest.skip("Icarus Verilog and vvp are required")
    spec = importlib.util.spec_from_file_location("qst6_oracle", RTL / "gen_qst6_round_vectors.py")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    rng = np.random.default_rng(2049)
    pairs = [(0x3fc00000, 0x3f800000), (0x7f800000, 0),
             (0x7f7fffff, 0x40000000), (0x00000001, 0x3f800000),
             (0x00ffffff, 0x3f000000), (0xbf000000, 0x3f800000)]
    for n in range(32):
        tie = int(np.array(np.float32(n + .5)).view(np.uint32))
        pairs.extend((bits | sign, 0x3f800000)
                     for bits in (tie - 1, tie, tie + 1) for sign in (0, 0x80000000))
    pairs += [tuple(map(int, row)) for row in rng.integers(0, 2**32, (1024, 2), dtype=np.uint32)]
    lines = []
    for a, b in pairs:
        a_ftz = (a & 0x80000000) if (a & 0x7f800000) == 0 else a
        b_ftz = (b & 0x80000000) if (b & 0x7f800000) == 0 else b
        x = np.array([a_ftz, b_ftz], dtype=np.uint32).view(np.float32)
        with np.errstate(all="ignore"):
            product = np.float32(x[0] * x[1])
        bits = int(np.array(product, dtype=np.float32).view(np.uint32))
        if (bits & 0x7f800000) == 0:
            bits &= 0x80000000
        invalid, q = oracle.ref(bits)
        if (a & 0x7f800000) == 0x7f800000 or (b & 0x7f800000) == 0x7f800000:
            invalid, q = 1, 0
        lines.append(f"{a:08x}{b:08x}{invalid:02x}{q:02x}\n")
    vec = tmp_path / "vectors.hex"
    vec.write_text("".join(lines), encoding="ascii")
    tb = tmp_path / "tb.sv"
    tb.write_text(f'''`timescale 1ns/1ps
module tb;
  reg clk=0, rst=1, iv=0, ready=0;
  reg [31:0] x=0, inv=0;
  wire ir, ov, invalid;
  wire signed [7:0] q;
  reg [79:0] v[0:{len(pairs)-1}];
  integer sent=0, received=0, cycle=0;
  reg pending=0, held=0, held_invalid=0;
  reg [7:0] held_q=0;
  always #5 clk=~clk;
  otpu_qst6_scale_round dut(clk,rst,iv,ir,x,inv,ov,ready,invalid,q);
  initial begin
    $readmemh("{vec.as_posix()}",v);
    repeat(2) @(negedge clk); rst=0;
    while (received < {len(pairs)} && cycle < 10000) begin
      @(negedge clk);
      iv=(sent < {len(pairs)}) && (((cycle % 5) != 0) || pending);
      if (iv) begin x=v[sent][79:48]; inv=v[sent][47:16]; end
      ready=(cycle % 7 != 0) && (cycle % 11 != 0);
      #1;
      if (held && (!ov || invalid !== held_invalid || q !== held_q))
        $fatal(1,"output changed under backpressure");
      held=ov && !ready; held_invalid=invalid; held_q=q;
      if (ov && ready) begin
        if (invalid !== v[received][8] || q !== v[received][7:0])
          $fatal(1,"mismatch %0d got %b %h expected %h",received,invalid,q,v[received]);
        received=received+1;
      end
      pending=iv && !ir;
      if (iv && ir) sent=sent+1;
      cycle=cycle+1;
    end
    if (received != {len(pairs)} || sent != received) $fatal(1,"timeout/lost data");
    @(negedge clk); x=32'h3f800000; inv=32'h3f800000; iv=1; ready=0;
    @(negedge clk); iv=0;
    repeat(3) @(negedge clk); #1;
    if (!ov) $fatal(1,"reset test did not produce a buffered output");
    rst=1; @(negedge clk); rst=0; #1;
    if (ov) $fatal(1,"reset did not discard pending output");
    $display("PASS: %0d multiply-round vectors with stalls",received); $finish;
  end
endmodule
''', encoding="ascii")
    exe = tmp_path / "scale_round.vvp"
    result = subprocess.run([iverilog, "-g2012", "-Wall", "-s", "tb", "-o", str(exe),
                             str(ROOT / "rtl" / "vpu" / "otpu_fp.sv"),
                             str(ROOT / "rtl" / "vpu" / "otpu_fpipe.sv"),
                             str(RTL / "otpu_qst6_scale_round.sv"), str(tb)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([vvp, str(exe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PASS: {len(pairs)} multiply-round vectors with stalls" in result.stdout


def test_qst6_scaled_rounding_against_independent_oracle(tmp_path):
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        pytest.skip("Icarus Verilog and vvp are required")
    (tmp_path / "build").mkdir()
    vectors = subprocess.run([sys.executable, str(RTL / "gen_qst6_round_vectors.py")],
                             cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert vectors.returncode == 0, vectors.stdout + vectors.stderr
    assert "10209 vectors" in vectors.stdout
    exe = tmp_path / "round.vvp"
    build = subprocess.run([iverilog, "-g2012", "-Wall", "-s", "tb_qst6_round",
                            "-o", str(exe), str(ROOT / "rtl" / "vpu" / "otpu_fp.sv"),
                            str(RTL / "otpu_qst6_round.sv"), str(RTL / "tb_qst6_round.sv")],
                           capture_output=True, text=True, timeout=30)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([vvp, str(exe)], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: 10209 independent binary32 vectors" in result.stdout

    # Cross-check the RTL rounding stage on values produced by the proposed
    # FP32 scaling reference. Scaling itself still runs in Python here.
    from opentpu import fp32 as F
    x = np.random.default_rng(614).normal(size=(32, 128)).astype(np.float32)
    expected, _ = quantize_rows(x)
    x = F.ftz(x)
    amax = F.from_bits(np.max(F.bits(F.fabs(x)), axis=1))
    scaled = F.mul(x, F.mul(np.float32(31), F.recip(amax))[:, None])
    (tmp_path / "build" / "qst6_round_vectors.txt").write_text(
        "".join(f"{int(bits):08x} 0 {int(value) & 255:02x}\n"
                for bits, value in zip(F.bits(scaled).flat, expected.flat)), encoding="ascii")
    result = subprocess.run([vvp, str(exe)], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: 4096 independent binary32 vectors" in result.stdout


def test_int6_codec_icarus_smoke(tmp_path):
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        pytest.skip("Icarus Verilog and vvp are required")
    exe = tmp_path / "codec.vvp"
    build = subprocess.run([iverilog, "-g2012", "-Wall", "-s", "tb_kv_int6_codec",
                            "-Ptb_kv_int6_codec.WORD_LIMIT=65536", "-o", str(exe),
                            str(RTL / "otpu_kv_int6_codec.sv"), str(RTL / "tb_kv_int6_codec.sv")],
                           capture_output=True, text=True, timeout=30)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([vvp, str(exe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: 65536 packed patterns and 1024 signed range cases" in result.stdout


def test_int6_stream_icarus_backpressure_and_boundaries(tmp_path):
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        pytest.skip("Icarus Verilog and vvp are required")
    exe = tmp_path / "stream.vvp"
    build = subprocess.run([iverilog, "-g2012", "-Wall", "-s", "tb_kv_int6_write_stream",
                            "-o", str(exe), str(RTL / "otpu_kv_int6_codec.sv"),
                            str(RTL / "otpu_kv_int6_write_stream.sv"),
                            str(RTL / "tb_kv_int6_write_stream.sv")],
                           capture_output=True, text=True, timeout=30)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([vvp, str(exe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: 266 groups, backpressure, byte/word/beat boundaries, rejects and reset" in result.stdout


def test_int6_full_word_stream_icarus(tmp_path):
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        pytest.skip("Icarus Verilog and vvp are required")
    exe = tmp_path / "word_stream.vvp"
    build = subprocess.run([iverilog, "-g2012", "-Wall", "-s", "tb_kv_int6_word_stream",
                            "-o", str(exe), str(RTL / "otpu_kv_int6_codec.sv"),
                            str(RTL / "otpu_kv_int6_word_stream.sv"),
                            str(RTL / "tb_kv_int6_word_stream.sv")],
                           capture_output=True, text=True, timeout=30)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([vvp, str(exe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: 256 groups, word/beat/page/4GiB edges, stalls, rejects, reset" in result.stdout


def test_int6_acknowledged_stream_icarus(tmp_path):
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        pytest.skip("Icarus Verilog and vvp are required")
    exe = tmp_path / "acked_stream.vvp"
    build = subprocess.run([iverilog, "-g2012", "-Wall", "-s", "tb_kv_int6_acked_stream",
                            "-o", str(exe), str(RTL / "otpu_kv_int6_codec.sv"),
                            str(RTL / "otpu_kv_int6_word_stream.sv"),
                            str(RTL / "otpu_kv_int6_acked_stream.sv"),
                            str(RTL / "tb_kv_int6_acked_stream.sv")],
                           capture_output=True, text=True, timeout=30)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([vvp, str(exe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: 128 randomized groups" in result.stdout


def test_int6_shared_idle_fence_icarus(tmp_path):
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        pytest.skip("Icarus Verilog and vvp are required")
    exe = tmp_path / "idle_stream.vvp"
    build = subprocess.run([iverilog, "-g2012", "-Wall", "-s", "tb_kv_int6_idle_stream",
                            "-o", str(exe), str(RTL / "otpu_kv_int6_codec.sv"),
                            str(RTL / "otpu_kv_int6_word_stream.sv"),
                            str(RTL / "otpu_kv_int6_idle_stream.sv"),
                            str(RTL / "tb_kv_int6_idle_stream.sv")],
                           capture_output=True, text=True, timeout=30)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([vvp, str(exe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: SW register stalls, shared idle fence" in result.stdout


def test_word_stream_matches_independent_quantized_reference(tmp_path):
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iverilog or not vvp:
        pytest.skip("Icarus Verilog and vvp are required")
    x = np.random.default_rng(789).normal(size=(64, 16)).astype(np.float32)
    q, _ = quantize_rows(x)
    packed = pack_rows(q)
    vectors = tmp_path / "values.hex"
    expected = tmp_path / "expected.hex"
    vectors.write_text("".join(f"{int.from_bytes(row.tobytes(), 'little'):032x}\n" for row in q), encoding="ascii")
    expected.write_text("".join(f"{int.from_bytes(row.tobytes(), 'little'):024x}\n" for row in packed), encoding="ascii")
    tb = tmp_path / "tb.sv"
    tb.write_text(f'''`timescale 1ns/1ps
module tb;
  reg clk=0, rst=1, iv=0, ready=0;
  reg [31:0] addr=0;
  reg [127:0] values=0;
  wire ir, reject, ov;
  wire [29:0] oa;
  wire [31:0] data;
  wire [3:0] be;
  reg [127:0] inputs[0:63];
  reg [95:0] expected[0:63];
  integer i,j;
  always #5 clk=~clk;
  otpu_kv_int6_word_stream dut(clk,rst,iv,ir,addr,values,reject,ov,ready,oa,data,be);
  initial begin
    $readmemh("{vectors.as_posix()}",inputs);
    $readmemh("{expected.as_posix()}",expected);
    @(negedge clk); rst=0; #1;
    for(i=0;i<64;i=i+1) begin
      if(!ir) $fatal(1,"input not ready");
      addr=i*16; values=inputs[i]; iv=1;
      @(negedge clk); iv=0;
      for(j=0;j<3;j=j+1) begin
        if(!ov || reject || oa !== ((i*16)>>2)+j || be !== 4'hf || data !== expected[i][j*32 +: 32])
          $fatal(1,"python mismatch row=%0d word=%0d",i,j);
        ready=1; @(negedge clk); ready=0;
      end
    end
    $display("PASS python reference: 64 rows"); $finish;
  end
endmodule
''', encoding="ascii")
    exe = tmp_path / "python_ref.vvp"
    build = subprocess.run([iverilog, "-g2012", "-Wall", "-s", "tb", "-o", str(exe),
                            str(RTL / "otpu_kv_int6_codec.sv"), str(RTL / "otpu_kv_int6_word_stream.sv"),
                            str(tb)], capture_output=True, text=True, timeout=30)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([vvp, str(exe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS python reference: 64 rows" in result.stdout
