// VOP RAND's generator, one per VPU lane (docs/isa.md, opentpu/rand.py): the counter-based
// uniform of a raw 32-bit counter word x under the key k,
//
//   bits = threefry2x32_13({x, 0}, {k, KEY1})[0]     (Random123's Threefry-2x32, 13 rounds)
//   u    = fp32((bits >> 8) * 2^-24)                 in [0, 1 - 2^-24], exact
//
// Additions, rotations and xors only (no DSP). Seven registered stages, the latency of a VPU
// multiply-add slot (1 + LM + LA = SL), so RAND issues and retires as ADD does: the lane's
// result mux takes `u` where it would take slot 0's sum. Every stage advances on `en` (the
// lane's enable); there is no reset (the pipeline holds data, the VPU's meta says when a word
// is live). Rounds per stage: init + 1 | 2 3 | 4 inj 5 | 6 7 | 8 inj 9 | 10 11 | 12 inj 13 +
// the conversion: at most three 32-bit additions on a path.
module otpu_rand #(
  parameter int NST = 7                       // stages: must be the VPU's SL
) (
  input  logic        clk,
  input  logic        en,
  input  logic [31:0] x,                      // the counter word (raw bits)
  input  logic [31:0] k,                      // the key (raw bits)
  output logic [31:0] u                       // the fp32 uniform, NST enabled cycles later
);
  localparam logic [31:0] KEY1 = 32'h6F70656E;
  localparam logic [31:0] PARITY = 32'h1BD11BDA;
  initial if (NST != 7) $fatal(1, "otpu_rand: 7 stages (the VPU's SL); NST %0d", NST);

  typedef struct packed {
    logic [31:0] x0, x1, k0;
  } st_t;

  function automatic logic [31:0] rotl(input logic [31:0] v, input int r);
    return (v << r) | (v >> (32 - r));
  endfunction
  // round r (0-based): X0 += X1; X1 = rotl(X1, R[r % 8]) ^ X0
  function automatic st_t rnd(input st_t s, input int r);
    st_t t;
    int rr;
    case (r % 8)
      0: rr = 13; 1: rr = 15; 2: rr = 26; 3: rr = 6;
      4: rr = 17; 5: rr = 29; 6: rr = 16; default: rr = 24;
    endcase
    t = s;
    t.x0 = s.x0 + s.x1;
    t.x1 = rotl(s.x1, rr) ^ t.x0;
    return t;
  endfunction
  // the key injection after round 4i: X0 += ks[i % 3]; X1 += ks[(i + 1) % 3] + i
  function automatic logic [31:0] ks(input logic [31:0] k0, input int j);
    case (j % 3)
      0: return k0;
      1: return KEY1;
      default: return PARITY ^ k0 ^ KEY1;
    endcase
  endfunction
  function automatic st_t inj(input st_t s, input int i);
    st_t t;
    t = s;
    t.x0 = s.x0 + ks(s.k0, i);
    t.x1 = s.x1 + ks(s.k0, i + 1) + 32'(i);
    return t;
  endfunction
  // the uniform of the top 24 bits: m * 2^-24 (m < 2^24 exactly representable)
  function automatic logic [31:0] to_u(input logic [31:0] b);
    logic [23:0] m;
    logic [23:0] n;
    int p;
    m = b[31:8];
    if (m == 0) return 32'd0;
    p = 0;
    for (int i = 0; i < 24; i++) if (m[i]) p = i;
    n = m << (23 - p);
    return {1'b0, 8'(127 - 24 + p), n[22:0]};
  endfunction

  st_t s0, s1, s2, s3, s4, s5;
  st_t c0, c1, c2, c3, c4, c5;
  always_comb begin
    st_t t;
    t = '{x0: x + k, x1: KEY1, k0: k};            // {x, 0} + {k, KEY1}
    c0 = rnd(t, 0);
    c1 = rnd(rnd(s0, 1), 2);
    c2 = rnd(inj(rnd(s1, 3), 1), 4);
    c3 = rnd(rnd(s2, 5), 6);
    c4 = rnd(inj(rnd(s3, 7), 2), 8);
    c5 = rnd(rnd(s4, 9), 10);
  end
  logic [31:0] c6;
  always_comb begin
    st_t t;
    t = rnd(inj(rnd(s5, 11), 3), 12);
    c6 = to_u(t.x0);
  end
  always_ff @(posedge clk) if (en) begin
    s0 <= c0; s1 <= c1; s2 <= c2; s3 <= c3; s4 <= c4; s5 <= c5;
    u <= c6;
  end
endmodule
