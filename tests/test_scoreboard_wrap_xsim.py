"""Vivado simulation of production footprint/conflict package, not the full sequencer."""
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
BIN = Path("C:/Xilinx/Vivado/2024.2/bin")


def test_production_footprint_wrapped_qst_and_dma(tmp_path):
    xvlog, xelab, xsim = (BIN / name for name in ("xvlog.bat", "xelab.bat", "xsim.bat"))
    if not all(path.is_file() for path in (xvlog, xelab, xsim)):
        pytest.skip("Vivado XSim 2024.2 is required")
    tb = tmp_path / "tb.sv"
    tb.write_text('''module tb;
 import otpu_pkg::*;
 cmd_t q, st, ld; fp_t fq, fs, fl; fpm_t product; fpo_t operands;
 logic [31:0] av[0:3], wv[0:3], got, seed, rows, kb; logic [55:0] exact;
 integer i,j;
 initial begin
  q='0; st='0; ld='0;
  // One QST row of D=32 elements; the final half wraps from 4 GiB to zero.
  q.op=OP_QST; q.flags=1; q.w1=0; q.w2=32'hfffffff0; q.w3=32'h800;
  q.w4=32'h00010001; q.w5=32'd32; q.w6=32'd32; q.w7=1;
  st.op=OP_ST; st.w1=0; st.w2=32'h200; st.w3=4;
  fq=footprint(q,32,1); fs=footprint(st,32,1);
  if(!conflict(fq,fs) || !conflict_dram(fq,fs) || !conflict(fs,fq))
    $fatal(1,"wrapped QST vs low ST not scored");
  // Exact end is conservatively widened too; the extra wait is safe.
  q.w2=32'hffffffe0; fq=footprint(q,32,1);
  if(!conflict(fq,fs)) $fatal(1,"exact-end QST not conservatively covered");
  st.w1=32'hffffffe0; fs=footprint(st,32,1);
  if(!conflict(fq,fs)) $fatal(1,"top-end QST conflict missed");
  // A wrapping DMA store must conflict with a load at the low end.
  st.w1=32'hfffffffc; st.w3=2; fs=footprint(st,32,1);
  ld.op=OP_LD; ld.w1=0; ld.w2=32'h400; ld.w3=1; fl=footprint(ld,32,1);
  if(!conflict(fs,fl)) $fatal(1,"wrapped ST vs low LD not scored");
  // QST's component spans can overflow before they reach mk(). Row 1 lands
  // at 0xf0 due to a negative stride represented modulo 2^32.
  q.w2=32'h100; q.w4=32'h00010002; q.w6=32'hfffffff0;
  st.w1=32'hf0; st.w3=4; fq=footprint(q,32,1); fs=footprint(st,32,1);
  if(!conflict(fq,fs)) $fatal(1,"QST span addition overflow not scored");
  // Two row-stride steps multiply to 2^32; the middle row is at 0x80000100.
  q.w4=32'h00010003; q.w6=32'h80000000; st.w1=32'h80000100;
  fq=footprint(q,32,1); fs=footprint(st,32,1);
  if(!conflict(fq,fs)) $fatal(1,"QST row-stride product overflow not scored");
  // Element-stride overflow can hide an intermediate address as well.
  q.w4=32'h00010001; q.w6=32'd32; q.w7=32'h9f767c46;
  st.w1=32'h9f767d44; fq=footprint(q,32,1); fs=footprint(st,32,1);
  if(!conflict(fq,fs)) $fatal(1,"QST element-stride product overflow not scored");
  // A wrapped source-row product must cover a DMA write to that source.
  q.w4=32'h00010003; q.w5=32'h80000000; q.w6=32'd32; q.w7=1;
  ld.w1=32'h50000000; ld.w2=32'h80000000; ld.w3=1;
  fq=footprint(q,32,1); fl=footprint(ld,32,1);
  if(!conflict(fq,fl)) $fatal(1,"QST source product overflow not scored");
  // The scale-record byte count exceeds 2^32 before range construction.
  q.flags=0; q.w1=0; q.w2=32'h10000000; q.w3=32'h1000; q.w4=32'hffffffff;
  q.w5=0; q.w6=32'd32; q.w7=1; st.w1=0; st.w2=32'h80000000; st.w3=1;
  fq=footprint(q,32,1); fs=footprint(st,32,1);
  if(!conflict(fq,fs)) $fatal(1,"QST scale span overflow not scored");
  // 2^30 words occupy 2^32 bytes: DMA range must not become empty.
  q='0; q.op=OP_QST; q.flags=1; q.w2=32'h10000; q.w3=32'h20000;
  q.w4=32'h00010001; q.w5=32; q.w6=32; q.w7=1;
  st='0; st.op=OP_ST; st.w1=32'h40000000; st.w2=32'h10000; st.w3=32'h40000000;
  ld=st; ld.op=OP_LD; fq=footprint(q,32,1); fs=footprint(st,32,1); fl=footprint(ld,32,1);
  if(!conflict(fq,fs) || !conflict(fq,fl)) $fatal(1,"DMA byte count overflow not scored");
  // LDX table row product overflows despite an ISA-valid aligned stride.
  q='0; q.op=OP_LDX; q.w1=32'h10000; q.w2=32'h400; q.w3=32'h800;
  q.w4=32'h00010001; q.w5=32'h00400000; q.w6=1; q.w7=1026;
  st='0; st.op=OP_ST; st.w1=32'h3fc10000; st.w2=32'h10000; st.w3=1;
  fq=footprint(q,128,1); fs=footprint(st,128,1);
  if(!conflict(fq,fs)) $fatal(1,"LDX table product overflow not scored");
  // LDX destination and QACT source may also overflow in their row strides.
  q.w4=32'h00010003; q.w5=128; q.w6=32'h80000000; q.w7=1;
  ld='0; ld.op=OP_LD; ld.w1=32'h50000000; ld.w2=32'h80000400; ld.w3=1;
  fq=footprint(q,128,1); fl=footprint(ld,128,1);
  if(!conflict(fq,fl)) $fatal(1,"LDX destination product overflow not scored");
  q='0; q.op=OP_QACT; q.w1=32'h400; q.w2=32'h00010003; q.w3=32'h80000000;
  fq=footprint(q,128,1);
  if(!conflict(fq,fl)) $fatal(1,"QACT source product overflow not scored");
  // MM weight rows and TMM GSC scale rows can overflow their stride products.
  q='0; q.op=OP_MM; q.flags=1; q.w1=32'h10000; q.w3=32'h400;
  q.w4=32'h00010003; q.w5=32'h80000000; q.w6=32'h00010003;
  st='0; st.op=OP_ST; st.w1=32'h80010000; st.w2=32'h200; st.w3=1;
  fq=footprint(q,128,1); fs=footprint(st,128,1);
  if(!conflict(fq,fs)) $fatal(1,"MM weight product overflow not scored");
  q.op=OP_TMM; q.flags=32; q.w2=32'h30000; q.w5=128; q.w7=32'h80000000;
  st.w1=32'h80030000; fq=footprint(q,128,1); fs=footprint(st,128,1);
  if(!conflict(fq,fs)) $fatal(1,"TMM scale product overflow not scored");
  // Product fits, but adding the final row's bytes wraps the span.
  q.op=OP_MM; q.flags=1; q.w2=0; q.w4=32'h00010002; q.w5=32'hffffff80; q.w7=0;
  st.w1=32'h0000ff80; fq=footprint(q,128,1); fs=footprint(st,128,1);
  if(!conflict(fq,fs)) $fatal(1,"MM weight span addition overflow not scored");
  q.flags=8'h80; q.w2=32'h30000; q.w5=128; q.w7=32'hfffffffc;
  st.w1=32'h0002fffc; fq=footprint(q,128,1); fs=footprint(st,128,1);
  if(!conflict(fq,fs)) $fatal(1,"MM scale span addition overflow not scored");
  // GATHER source and destination products, including a multi-slice segment.
  q='0; q.op=OP_GATHER; q.w1=32'h400; q.w2=32'h1000; q.w3=32'h00010003;
  q.w4=32'h80000000; q.w5=1; q.w6=1;
  ld='0; ld.op=OP_LD; ld.w1=32'h50000000; ld.w2=32'h80000400; ld.w3=1;
  fq=footprint(q,128,1); fl=footprint(ld,128,1);
  if(!conflict(fq,fl)) $fatal(1,"GATHER source product overflow not scored");
  q.w1=32'h1000; q.w2=32'h400; q.w4=1; q.w5=32'h80000000;
  fq=footprint(q,128,1);
  if(!conflict(fq,fl)) $fatal(1,"GATHER destination product overflow not scored");
  q.w2=32'h100; q.w3=32'h00200002; q.w4=32; q.w5=32'hfffffff0;
  ld.w2=32'hf0; fq=footprint(q,128,1); fl=footprint(ld,128,1);
  if(!conflict(fq,fl)) $fatal(1,"GATHER destination sum overflow not scored");
  q.w2=32'h400; q.w3=32'h00010001; q.w4=1; q.w5=1; q.w6=32'h80000000;
  ld.w2=32'h80000400; fq=footprint(q,128,3); fl=footprint(ld,128,3);
  if(!conflict(fq,fl)) $fatal(1,"GATHER segment product overflow not scored");
  // DSTEP's two scale words can wrap when gs = 0xffffffff.
  q='0; q.op=OP_DSTEP; q.w1=32'h10000; q.w2=32'h200; q.w3=32'h300;
  q.w4=32'h00400001; q.w5=32'h400; q.w6=32'h500; q.w7=32'hffffffff;
  st='0; st.op=OP_VOP; st.w1=32'h3ff; st.w2=32'h900; st.w4=32'h00010001;
  st.w5=32'h00010001; st.w6=(32'(V_COPY)<<16)|1;
  fq=footprint(q,128,1); fs=footprint(st,128,1);
  if(!conflict(fq,fs)) $fatal(1,"DSTEP scale offset overflow not scored");
  // Cross-check all four product low words and overflow flags against a
  // full-width multiplication for varied rows, KB and strides.
  seed=32'h8a6e7311; q='0; q.op=OP_QST;
  for(i=0;i<2048;i=i+1) begin
    seed=seed*32'd1664525+32'd1013904223; rows=(seed & 16'hffff) | 1;
    seed=seed*32'd1664525+32'd1013904223; kb=(seed & 16'hffff) | 1;
    q.w4={kb[15:0],rows[15:0]};
    seed=seed*32'd1664525+32'd1013904223; q.w5=seed;
    seed=seed*32'd1664525+32'd1013904223; q.w6=seed;
    seed=seed*32'd1664525+32'd1013904223; q.w7=seed;
    product=fp_prod(q,128,1);
    av[0]=rows-1; av[1]=rows-1; av[2]=kb*128-1; av[3]=rows;
    wv[0]=q.w5; wv[1]=q.w6; wv[2]=q.w7; wv[3]=1+((kb+3)>>2);
    for(j=0;j<4;j=j+1) begin
      exact=56'(av[j])*56'(wv[j]);
      case(j) 0: got=product.p0; 1: got=product.p1; 2: got=product.p2; 3: got=product.p3; endcase
      if(got!==exact[31:0] || product.ov[j] !== (|exact[55:32]))
        $fatal(1,"product mismatch case=%0d lane=%0d",i,j);
    end
  end
  // The split product helper is shared by other instruction footprints.
  for(i=0;i<1024;i=i+1) begin
    operands='0;
    for(j=0;j<4;j=j+1) begin
      seed=seed*32'd1664525+32'd1013904223; operands.a[j]=seed[23:0];
      seed=seed*32'd1664525+32'd1013904223; operands.w[j]=seed;
    end
    product=fp_sum(fp_part(operands));
    for(j=0;j<4;j=j+1) begin
      exact=56'(operands.a[j])*56'(operands.w[j]);
      case(j) 0: got=product.p0; 1: got=product.p1; 2: got=product.p2; 3: got=product.p3; endcase
      if(got!==exact[31:0] || product.ov[j] !== (|exact[55:32]))
        $fatal(1,"generic product mismatch case=%0d lane=%0d",i,j);
    end
  end
  // Boundary and carry from adding two individually in-range partial products.
  operands='0; operands.a[0]=24'd65535; operands.w[0]=32'd65537;
  product=fp_sum(fp_part(operands));
  if(product.p0!==32'hffffffff || product.ov[0]!==1'b0) $fatal(1,"product exact top");
  operands.w[0]=32'd65538; product=fp_sum(fp_part(operands));
  if(product.p0!==32'h0000fffe || product.ov[0]!==1'b1) $fatal(1,"product partial-sum carry");
  $display("PASS: production QST and DMA wrapped footprints"); $finish;
 end
endmodule
''', encoding="ascii")
    for command in ([str(xvlog), "-sv", str(ROOT / "rtl/top/otpu_pkg.sv"), str(tb)],
                    [str(xelab), "tb", "-s", "scoreboard_wrap_snapshot"],
                    [str(xsim), "scoreboard_wrap_snapshot", "-runall"]):
        result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, f"{command[0]} exit {result.returncode}: {result.stdout}\n{result.stderr}"
    assert "PASS: production QST and DMA wrapped footprints" in result.stdout
