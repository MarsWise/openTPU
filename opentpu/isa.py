"""openTPU instruction encoding (docs/isa.md). Every instruction is eight 32-bit words."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NOP, HALT, LI, ADDI, LOOP, BAR, RLD, WAITW = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07
LD, ST, DSTEP, STREAM, LDX = 0x10, 0x11, 0x12, 0x13, 0x14
MM, QACT, QST, TMM = 0x20, 0x21, 0x22, 0x23
VOP = 0x30
GATHER = 0x40

OPNAMES = {NOP: "NOP", HALT: "HALT", LI: "LI", ADDI: "ADDI", LOOP: "LOOP", BAR: "BAR", RLD: "RLD",
           WAITW: "WAITW",
           LD: "LD", ST: "ST", LDX: "LDX", DSTEP: "DSTEP", STREAM: "STREAM", MM: "MM", QACT: "QACT",
           QST: "QST", TMM: "TMM", VOP: "VOP", GATHER: "GATHER"}

# MM / QACT / QST flags
F_UNIT, F_ACC, F_RMAX, F_ASCALE = 0x1, 0x2, 0x4, 0x8     # MM
# MM weight format, flags[5:4] (docs/isa.md "Weight formats")
WF_SHIFT = 4
# int8 | int4 / E2M1 with a 6-bit multiplier per D/4 | E2M1 with an E4M3 scale per D/8; every
# format a per-row fp32 scale Sw and power-of-two offsets dw under it (MX, docs/mx_mxu.md)
W8, W4I, W4F, WNV = 0, 1, 2, 3
WFORMATS = {"int8": W8, "int4": W4I, "fp4": W4F, "nvfp4": WNV}
MX_W = {W8: 16, W4I: 11, W4F: 11, WNV: 13}   # the shift window W of each format
F_PAIR = 0x40                   # MM, 4-bit only: column reuse, two K-blocks per cycle and row;
#                                 TMM PAIR (Config.TPAIR): the same, a ternary chunk in two advances
F_RSC = 0x80                    # MM: row scales only (each scale row is Sw[n]; dw 0, m 1)
F_ROW, F_CSCALE, F_RSCALE = 0x1, 0x2, 0x4   # QACT (QST: F_ROW)
F_DUP = 0x8                     # QACT: also write the rows to ACT rows rows..2*rows-1 (PAIR)
F_DUP4 = 0x10                   # QACT, with F_DUP: and to rows 2*rows..4*rows-1 (TMM QUAD)
F_QUAD = 0x10                   # TMM (its weight-format field's bit4; Config.TQUAD): the K-split
#                                 by four, a ternary chunk's four blocks in one advance
F_GSC = 0x20                    # TMM: fp32 scale per row and D-element K block (Config.TGSC)
TMM_BIN = 1 << 26               # TMM word 4: 8 binary weights per byte (Config.TBIN)
TMM_GRQ = 1 << 28               # TMM word 4, GSC: block-major packed layout (Config.TGRQ)
TMM_CQ = 1 << 27                # TMM word 4, GSC: codebook codes, each the LUT's int8 (Config.TCQ):
#                                 2-bit (the ternary layout), with BIN 1-bit, with CQ4 4-bit
TMM_CQ4 = 1 << 29               # TMM word 4, CQ: 4-bit codes, two D-element blocks a D-byte chunk
TMM_LUT = 1 << 30               # TMM word 4: LUTSET, eight of the codebook LUT's 16 int8 entries
TMM_T5 = TMM_BIN | TMM_CQ4      # TMM word 4 (the pair, CQ clear): dense trits, five a byte, five
#                                 D-element blocks a D-byte chunk (opentpu/trits.py; Config.TD5)
F_HALF = 0x2                    # QST, ROW mode: write only the first half of each row
F_Q4 = 0x4                      # QST, ROW mode, es 1: 4-bit rows (int4, S = amax/7; Config.QST4)
F_RAW = 0x1                     # RLD: the word's bits (no fp32 -> int conversion)
F_MUL = 0x2                     # RLD: times R[rb] + w2 (mod 2^32)
F_BOUND = 0x4                   # RLD: w3 <= the value <= w4 (the waits pass's; the hardware ignores it)
F_CHAIN = 0x1                   # HALT: then load and start the program at R[ra], R[rb] words
C_EQ, C_NE, C_GE = 0, 1, 2      # WAITW flags[1:0]: the word ==, != or >= (signed difference) ref

# VOP functions
V_ADD, V_SUB, V_RSUB, V_MUL, V_MAX, V_MIN, V_OUTER = 0, 1, 2, 3, 4, 5, 6
V_MAC = 7                 # A * Cv(c) + B: Cv the column vector at imm (TMEM, w7 += R[rd])
V_COPY, V_EXP2, V_RECIP, V_RSQRT, V_ABS, V_FILL, V_EXP2SUB, V_LOG2 = 8, 9, 10, 11, 12, 13, 14, 15
V_RSUM, V_RMAX, V_RSSQ, V_RDOT, V_ARGMAX = 16, 17, 18, 19, 20
V_CMP_GT, V_CMP_EQ, V_SEL = 21, 22, 23
V_U8_0, V_U8_1, V_U8_2, V_U8_3 = 24, 25, 26, 27
V_U8 = (V_U8_0, V_U8_1, V_U8_2, V_U8_3)
V_RAND = 28               # uniforms in [0, 1): Threefry-2x32-13 of A's raw word under the key
#                           imm = R[rd] + w7 (raw bits; opentpu/rand.py; Config.VRAND)
VFUNCS = {V_ADD: "add", V_SUB: "sub", V_RSUB: "rsub", V_MUL: "mul", V_MAX: "max",
          V_MIN: "min", V_OUTER: "outer", V_MAC: "mac", V_COPY: "copy", V_EXP2: "exp2", V_RECIP: "recip",
          V_RSQRT: "rsqrt", V_ABS: "abs", V_FILL: "fill", V_EXP2SUB: "exp2sub",
          V_LOG2: "log2", V_RSUM: "rsum", V_RMAX: "rmax", V_RSSQ: "rssq", V_RDOT: "rdot",
          V_ARGMAX: "argmax", V_CMP_GT: "cmp_gt", V_CMP_EQ: "cmp_eq", V_SEL: "sel"}
VFUNCS.update({f: f"u8_{i}" for i, f in enumerate(V_U8)})
VFUNCS[V_RAND] = "rand"
V_BFLY = 29               # the Hadamard butterfly stage at stride s = imm (u32, a power of two,
#                           cols % 2s == 0): B is A's tile (b = a, brs = ars) at column c ^ s;
#                           dst(r, c) = A(r, c) + B(r, c) if c & s == 0, else B(r, c) - A(r, c)
#                           (Config.VBFLY; dst must not overlap A)
VFUNCS[V_BFLY] = "bfly"
V_AFF = 30                # the fused row/column affine (Config.VAFF):
#                           dst(r, c) = (A(r, c) * P(r)) * G(c) + H(c), P per row (B ROW),
#                           G(c) = T[imm + c], H(c) = T[imm + cols + c] (-0 with flag DONE);
#                           cols <= 256 (the column buffers), G and H read first
VFUNCS[V_AFF] = "aff"
# TFN, the table functions (opentpu/tfn.py; Config.TFN, CAPS2 bit3): elementwise on A
V_SIGMOID, V_SILU, V_TANH, V_GELU, V_GELU_TANH, V_SOFTPLUS, V_MISH, V_QGELU, V_SIN, V_COS, \
    V_ATAN = range(0x20, 0x2B)          # V_ATAN: atan on [0, 1] (Config.TATAN, CAPS2 bit12)
VFUNCS.update({V_SIGMOID: "sigmoid", V_SILU: "silu", V_TANH: "tanh", V_GELU: "gelu",
               V_GELU_TANH: "gelu_tanh", V_SOFTPLUS: "softplus", V_MISH: "mish",
               V_QGELU: "quick_gelu", V_SIN: "sin", V_COS: "cos", V_ATAN: "atan"})
TFN_FUNCS = set(range(0x20, 0x2B))
BINARY = {V_ADD, V_SUB, V_RSUB, V_MUL, V_MAX, V_MIN, V_FILL, V_EXP2SUB, V_MAC,
          V_CMP_GT, V_CMP_EQ, V_SEL, V_BFLY}
REDUCE = {V_RSUM, V_RMAX, V_RSSQ, V_RDOT, V_ARGMAX}
READS_B = BINARY | {V_RDOT}          # functions that read operand B in its bmode (OUTER: B_ROW)
F_DSCALAR, F_DONE = 0x1, 0x2         # VOP OUTER: decay T[d] for every column / decay 1.0
OUTER_MAX_COLS = 256                 # OUTER: its column vectors are held in 256-word buffers

# VOP broadcast modes for operand B
B_FULL, B_ROW, B_COL, B_SCALAR = 0, 1, 2, 3


def u32(x: int) -> int:
    return int(x) & 0xFFFFFFFF


def f32bits(x: float) -> int:
    return int(np.asarray(x, dtype=np.float32).view(np.uint32))


@dataclass
class Instr:
    op: int
    ra: int = 0
    rb: int = 0
    rc: int = 0
    rd: int = 0
    flags: int = 0
    w: list = field(default_factory=lambda: [0] * 7)   # w1..w7
    comment: str = ""
    src: tuple = ()          # kernel source frames that emitted it (set by the compiler)

    def encode(self) -> list[int]:
        _check(all(0 <= r < 16 for r in (self.ra, self.rb, self.rc, self.rd)),
               f"{OPNAMES.get(self.op, self.op)}: a register field outside R0..R15")
        _check(0 <= self.flags < 256, f"{OPNAMES.get(self.op, self.op)}: flags {self.flags:#x}")
        w0 = (self.op & 0xFF) | (self.ra << 8) | (self.rb << 12) | (self.rc << 16) \
            | (self.rd << 20) | ((self.flags & 0xFF) << 24)
        return [u32(w0)] + [u32(x) for x in self.w]

    @staticmethod
    def decode(words) -> "Instr":
        w0 = int(words[0])
        return Instr(op=w0 & 0xFF, ra=(w0 >> 8) & 15, rb=(w0 >> 12) & 15, rc=(w0 >> 16) & 15,
                     rd=(w0 >> 20) & 15, flags=(w0 >> 24) & 0xFF,
                     w=[int(x) for x in words[1:8]])

    def __str__(self) -> str:
        name = OPNAMES.get(self.op, f"op{self.op:#x}")
        regs = f"ra=R{self.ra} rb=R{self.rb} rc=R{self.rc} rd=R{self.rd}"
        c = f"  ; {self.comment}" if self.comment else ""
        return f"{name:6s} {regs} fl={self.flags:#x} w={[hex(x) for x in self.w]}{c}"


def _check(ok: bool, what: str) -> None:
    """A field that does not fit its bits, or a flag combination the instruction does not
    have: an error (an assert would let python -O truncate the word silently)."""
    if not ok:
        raise ValueError(what)


def _w(*vals) -> list:
    v = [u32(x) for x in vals]
    return v + [0] * (7 - len(v))


def nop(comment=""):
    return Instr(NOP, comment=comment)


def halt(chain=False, ra=0, rb=0, comment=""):
    """HALT; with `chain`, the board then loads R[rb] instructions from DRAM byte address R[ra]
    into IMEM and starts them (docs/isa.md "HALT CHAIN")."""
    return Instr(HALT, ra=ra, rb=rb, flags=F_CHAIN if chain else 0, comment=comment)


def li(rd, imm, comment=""):
    return Instr(LI, rd=rd, w=_w(imm), comment=comment)


def addi(rd, ra, imm, comment=""):
    return Instr(ADDI, rd=rd, ra=ra, w=_w(imm), comment=comment)


def loop(body_len, count, rcount=0, comment=""):
    return Instr(LOOP, ra=rcount, w=_w(body_len, count), comment=comment)


def bar():
    return Instr(BAR)


def rld(rd, tmem, ra=0, raw=False, mul=None, rb=0, bound=None, comment=""):
    """R[rd] = f2i(T[R[ra] + tmem]) (raw: the word's bits), times R[rb] + mul (mod 2^32) if mul
    is given; docs/isa.md "RLD". bound (lo, hi): the values the register may take, lo <= v <= hi
    as unsigned 32-bit numbers, what the kernel knows of the word (an index below E, a slot of
    the pool): the waits pass bounds the footprints computed from the register by it
    (opentpu.waits), the ISA simulator checks it, the hardware ignores it (flag bit 2, w3, w4)."""
    flags = (F_RAW if raw else 0) | (F_MUL if mul is not None else 0)
    lo = hi = 0
    if bound is not None:
        lo, hi = (int(x) for x in bound)
        _check(0 <= lo <= hi <= 0xFFFFFFFF, f"RLD: bound {bound}")
        flags |= F_BOUND
    return Instr(RLD, ra=ra, rb=rb, rd=rd, flags=flags,
                 w=_w(tmem, 0 if mul is None else mul, lo, hi), comment=comment)


def waitw(dram, tmem, ref, cmp, ra=0, rb=0, rc=0, mask=0xFFFFFFFF, interval=0, timeout=0,
          comment=""):
    """WAITW (DMA): wait until cmp(M32[R[ra] + dram] & mask, R[rc] + ref) holds, polling DRAM
    every `interval` cycles (the first read at once; timeout cycles, 0: none), then
    T[R[rb] + tmem] = the word's bits (an RLD raw takes it into a register); docs/isa.md
    "WAITW"."""
    if cmp not in (C_EQ, C_NE, C_GE):
        raise ValueError(f"WAITW compare {cmp}")
    return Instr(WAITW, ra=ra, rb=rb, rc=rc, flags=cmp,
                 w=_w(dram, tmem, ref, mask, interval, timeout), comment=comment)


def waitw_holds(word: int, ref: int, cmp: int, mask: int = 0xFFFFFFFF) -> bool:
    """WAITW's condition on the word it read."""
    v, ref = u32(word) & u32(mask), u32(ref)
    if cmp == C_EQ:
        return v == ref
    if cmp == C_NE:
        return v != ref
    return u32(v - ref) < 1 << 31               # GE: the difference >= 0 as signed


def f2i(bits: int) -> int:
    """RLD's conversion of fp32 bits to a 32-bit word: truncation toward zero; values below 1 in
    magnitude (denormals, +-0) give 0, and |x| >= 2^31, infinities and NaN give 0x80000000."""
    bits = u32(bits)
    e = (bits >> 23) & 0xFF
    if e < 127:
        return 0
    if e >= 127 + 31:
        return 0x80000000
    m = (bits & 0x7FFFFF) | 0x800000
    v = m << (e - 150) if e >= 150 else m >> (150 - e)
    return u32(-v if bits >> 31 else v)


def ld(dram, tmem, nwords, ra=0, rb=0, comment=""):
    return Instr(LD, ra=ra, rb=rb, w=_w(dram, tmem, nwords), comment=comment)


LDX_MAX_COLS = 4096                  # LDX: words per row
LDX_MAX_ROWS_E = 1 << 24             # LDX: table rows E


def ldx(dram, tmem, idx, rows, cols, srs, drs, nrows, raw=False, ra=0, rb=0, rc=0, D=128,
        comment=""):
    """LDX (DMA, Config.LDX, CAPS2 bit14): the indexed row gather from DRAM (docs/isa.md "LDX").
    For i < rows: j = clamp(f2i(T[R[rc] + idx + i]), 0, nrows - 1) (raw: the word as a signed
    integer), T[R[rb] + tmem + i*drs + c] = M32[R[ra] + dram + j*srs + 4c] for c < cols. The
    table's rows start on D-byte chunks: dram (with R[ra]) and srs multiples of D. Deformable
    attention's samples, embedding rows, MoE expert rows."""
    _check(0 < rows < 65536 and 0 < cols <= LDX_MAX_COLS,
           f"LDX: rows {rows} (1..65535), cols {cols} (1..{LDX_MAX_COLS})")
    _check(0 < nrows <= LDX_MAX_ROWS_E, f"LDX: table rows {nrows} (1..2^24)")
    _check(srs % D == 0 and 0 <= srs // D < 65536 and dram % D == 0,
           f"LDX: the table base {dram} and row stride {srs} must be multiples of D = {D} "
           "(the stride below 65536 chunks)")
    return Instr(LDX, ra=ra, rb=rb, rc=rc, flags=F_RAW if raw else 0,
                 w=_w(dram, tmem, idx, rows | (cols << 16), srs, drs, nrows), comment=comment)


def st(dram, tmem, nwords, ra=0, rb=0, comment=""):
    return Instr(ST, ra=ra, rb=rb, w=_w(dram, tmem, nwords), comment=comment)


DSTEP_LANES = 8                      # DSTEP: state words per cycle (timing only)
DSTEP_MAX_ROWS = 256                 # DSTEP: o is held in a 256-word buffer
F_DZERO = 0x1                        # DSTEP: the state starts at +0 (DRAM is not read)


def dstep(dram, qk, v, rows, cols, g, gs, o, zero=False, ra=0, rb=0, rc=0, comment=""):
    """DSTEP (DMA): one Gated DeltaNet head step on an fp32 state St [rows, cols] (row-major)
    in DRAM at byte address R[ra] + dram, updated in place. q = T[qk + c], k = T[qk + cols + c]
    (qk: R[rb] + w2), v(r) = T[v + r] (R[rc] + w3), the decay e = T[g], beta = T[g + gs], and
    o(r) = T[o + r] is written. Row by row: kv = rdot(St[r], k), d = (v(r) - kv * e) * beta,
    St[r] = St[r] * e + d * k, o(r) = rdot(St[r], q) -- the rounding of RDOT, MUL, SUB, MUL,
    OUTER, RDOT (docs/isa.md). `zero`: the state starts at +0 and is not read (position 0)."""
    _check(0 < rows < 65536 and 0 < cols < 65536, f"DSTEP: rows {rows}, cols {cols}")
    return Instr(DSTEP, ra=ra, rb=rb, rc=rc, flags=F_DZERO if zero else 0,
                 w=_w(dram, qk, v, rows | (cols << 16), g, o, gs), comment=comment)


# ------------------------------------------------------------------------------------ STREAM
# The stream engine's instruction (docs/stream.md): rows of a stream (DRAM or TMEM) pass
# REDUCE A -> SCALAR -> MAP U -> MAP/REDUCE Q under a descriptor of a few TMEM words.
F_SZERO = 0x1                        # the stream reads as +0 (nothing is read)
F_SRC_T = 0x2                        # the stream source is TMEM (words), not DRAM (bytes)
F_DST_T = 0x4                        # the stream destination is TMEM
F_NODST = 0x8                        # the stream is not written back

A_SLOT, A_SELF, A_CONST, A_MAX = 0, 1, 2, 3      # REDUCE A: A = isum_64(S[r] * opA), or
#                                                   the row's max (A_MAX, VOP RMAX's order)
U_PASS, U_FMMA, U_MUL, U_ADD = 0, 1, 2, 3         # MAP U: Y = S | S*G + D*B | S*G | S + D*B
G_REG, G_SLOT, G_CONST, G_ONE = 0, 1, 2, 3        # G: register, column slot, constant, 1.0
B_SLOT, B_CONST, B_REG = 0, 1, 2                  # B: column slot, constant, register
# SCALAR ops: dst = op(a, b), each rounded as fp32.py's function of the same name
SC_ADD, SC_SUB, SC_MUL, SC_MAX, SC_MIN, SC_MOV = 0, 1, 2, 3, 4, 5
SC_RSQRT, SC_RECIP, SC_EXP2, SC_LOG2, SC_EXP2SUB = 6, 7, 8, 9, 10
# SCALAR operands: r0..r7, the row's A, its row scalar X[r], constants K0..K3, +0, +1
OP_A, OP_X, OP_K0, OP_ZERO, OP_ONE = 8, 9, 10, 14, 15
STREAM_SLOTS = 4                     # column slot i at vec + i * cols
STREAM_MAX_OPS = 15
STREAM_DESC_FIXED = 5                # descriptor words before the scalar ops
STREAM_DESC_AREA = 64                # the compiler's descriptor words at the top of TMEM


def _fsafe(p: int) -> int:
    """A 24-bit payload as a float-safe word: exponent field 0x80 (a normal fp32 in [2, 4)),
    payload in the sign bit and the 23 mantissa bits, so a FILL writes it exactly."""
    _check(0 <= p < 1 << 24, f"STREAM descriptor payload {p:#x} is not 24 bits")
    return ((p >> 23) & 1) << 31 | 0x80 << 23 | (p & 0x7FFFFF)


def _payload(w: int) -> int:
    w = int(w) & 0xFFFFFFFF
    if (w >> 23) & 0xFF != 0x80:
        raise ValueError(f"STREAM descriptor word {w:#010x} is not float-safe")
    return ((w >> 31) & 1) << 23 | (w & 0x7FFFFF)


@dataclass(frozen=True)
class StreamDesc:
    """A STREAM descriptor (docs/stream.md, section 3.2). Per row r of the stream S [rows, cols]:
      A     = isum_64(S[r] * opA)                    (a_en; opA: slot a_idx, S itself, K[a_idx];
                                                      A_MAX: the row's max)
      regs  : the scalar ops in order, over r0..r7, A, X[r], K0..K3, +0, +1
      Y     = S | S*G + D*B | S*G | S + D*B          (u_mode; D = r[d_reg])
      O[r]  = isum_64(Y * Q) (q_en; Q = slot q_idx, or Y with q_self), or r[o_reg] (o_reg_en)
      the stream row written back is Y, or Y * C[q_idx] with out_p2
    r4..r7 start as K0..K3 where rinit has their bit (else +0) and carry from row to row; r0..r3
    start each instruction at +0 and also carry. rsave writes r4..r7 back to K0..K3's words."""
    rows: int
    cols: int
    a_en: bool = False
    a_op: int = A_SLOT
    a_idx: int = 0
    u_mode: int = U_PASS
    g_src: int = G_ONE
    g_idx: int = 0
    b_src: int = B_SLOT
    b_idx: int = 0
    d_reg: int = 0
    q_en: bool = False
    q_idx: int = 0
    q_self: bool = False
    out_p2: bool = False
    o_reg_en: bool = False
    o_reg: int = 0
    rinit: int = 0
    rsave: int = 0
    srs: int = 0                     # TMEM stream row strides (0: cols)
    drs: int = 0
    ops: tuple = ()                  # ((op, dst, a, b), ...)

    def words(self) -> list[int]:
        _check(0 < self.rows < 1 << 12 and 0 < self.cols < 1 << 12,
               f"STREAM descriptor: rows {self.rows}, cols {self.cols} (1 .. 4095)")
        _check(len(self.ops) <= STREAM_MAX_OPS, f"STREAM descriptor: {len(self.ops)} ops")
        _check(0 <= self.srs < 1 << 12 and 0 <= self.drs < 1 << 12,
               f"STREAM descriptor: srs {self.srs}, drs {self.drs} (0 .. 4095)")
        c1 = (int(self.a_en) | self.a_op << 1 | self.a_idx << 3 | self.u_mode << 5
              | self.g_src << 7 | self.g_idx << 9 | self.b_src << 12 | self.b_idx << 14
              | self.d_reg << 17 | int(self.q_en) << 20 | self.q_idx << 21
              | int(self.out_p2) << 23)
        c2 = (len(self.ops) | self.rinit << 4 | self.rsave << 8 | int(self.q_self) << 12
              | int(self.o_reg_en) << 13 | self.o_reg << 14)
        p = [self.rows | self.cols << 12, c1, c2, 0, self.srs | self.drs << 12]   # d3 reserved
        p += [op | dst << 4 | a << 7 | b << 11 for op, dst, a, b in self.ops]
        return [_fsafe(x) for x in p]

    @staticmethod
    def decode(words) -> "StreamDesc":
        p = [_payload(w) for w in words[:STREAM_DESC_FIXED]]
        c1, c2 = p[1], p[2]
        n = c2 & 15
        if len(words) < STREAM_DESC_FIXED + n:
            raise ValueError("STREAM descriptor: fewer words than its ops")
        ops = tuple((x & 15, (x >> 4) & 7, (x >> 7) & 15, (x >> 11) & 15)
                    for x in (_payload(w) for w in
                              words[STREAM_DESC_FIXED:STREAM_DESC_FIXED + n]))
        return StreamDesc(rows=p[0] & 0xFFF, cols=p[0] >> 12, a_en=bool(c1 & 1),
                          a_op=(c1 >> 1) & 3, a_idx=(c1 >> 3) & 3, u_mode=(c1 >> 5) & 3,
                          g_src=(c1 >> 7) & 3, g_idx=(c1 >> 9) & 7, b_src=(c1 >> 12) & 3,
                          b_idx=(c1 >> 14) & 7, d_reg=(c1 >> 17) & 7,
                          q_en=bool((c1 >> 20) & 1), q_idx=(c1 >> 21) & 3,
                          out_p2=bool((c1 >> 23) & 1), rinit=(c2 >> 4) & 15,
                          rsave=(c2 >> 8) & 15, q_self=bool((c2 >> 12) & 1),
                          o_reg_en=bool((c2 >> 13) & 1), o_reg=(c2 >> 14) & 7,
                          srs=p[4] & 0xFFF, drs=p[4] >> 12, ops=ops)

    def nwords(self) -> int:
        return STREAM_DESC_FIXED + len(self.ops)


def stream(desc, src, dst, vec, x, k, out, ks=1, zero=False, src_t=False, dst_t=False,
           nodst=False, ra=0, rb=0, rc=0, rd=0, comment=""):
    """STREAM (docs/stream.md): the descriptor at TMEM `desc` over the rows of the stream at
    src + R[ra] (DRAM bytes, or TMEM words with src_t), Y written to dst + R[ra]; column slots
    at vec + R[rb] (slot i at + i * cols), row scalars X[r] = T[x + R[rc] + r], constants
    K_j = T[k + R[rd] + j * ks], row outputs O[r] = T[out + r]. desc and ks share w1 (desc in
    [15:0], ks in [31:16]) so the scoreboard knows the constants' range."""
    _check(0 <= desc < 1 << 16 and 0 <= ks < 1 << 16, f"STREAM: desc {desc}, ks {ks} (16 bits)")
    fl = (F_SZERO if zero else 0) | (F_SRC_T if src_t else 0) | (F_DST_T if dst_t else 0) | \
        (F_NODST if nodst else 0)
    return Instr(STREAM, ra=ra, rb=rb, rc=rc, rd=rd, flags=fl,
                 w=_w(desc | ks << 16, src, dst, vec, x, k, out), comment=comment)


# the scalar programs of the hardware's d stage (docs/stream.md, 4.4), d in r2
D_DELTA = ((SC_MUL, 0, OP_A, OP_K0), (SC_SUB, 1, OP_X, 0), (SC_MUL, 2, 1, OP_K0 + 1))
D_DELTA1 = ((SC_SUB, 1, OP_X, OP_A), (SC_MUL, 2, 1, OP_K0 + 1))
D_SCALE = ((SC_MUL, 2, OP_X, OP_K0 + 1),)
D_DOT = ((SC_MUL, 2, OP_A, OP_K0 + 1),)
DMODES = {D_DELTA: 0, D_DELTA1: 1, D_SCALE: 2, D_DOT: 3}


def state_desc(rows, cols, dprog=D_DELTA, a_idx=1, gate="const", q=True) -> StreamDesc:
    """A linear-recurrence step (docs/stream.md, 5.2): slots q (0), b = k (1), a (2), gate
    column (3); X = the rows' inputs (v); K0 = the scalar gate, K1 = beta.
    dprog: D_DELTA / D_DELTA1 / D_SCALE / D_DOT; a_idx: dot A's slot (1 or 2; unused unless
    dprog reads A); gate: "const" (K0), "col" (slot 3) or "one"."""
    uses_a = any(OP_A in (a, b) for _, _, a, b in dprog)
    g_src, g_idx = {"const": (G_CONST, 0), "col": (G_SLOT, 3), "one": (G_ONE, 0)}[gate]
    return StreamDesc(rows, cols, a_en=uses_a, a_op=A_SLOT, a_idx=a_idx if uses_a else 0,
                      u_mode=U_FMMA, g_src=g_src, g_idx=g_idx, b_src=B_SLOT, b_idx=1,
                      d_reg=2, q_en=q, q_idx=0, ops=tuple(dprog))


def gdn_desc(rows, cols) -> StreamDesc:
    """Gated DeltaNet, exactly DSTEP (ks = DSTEP's gs): slots q (0), k (1); X = v; K0 = the
    decay e, K1 = beta."""
    return state_desc(rows, cols, D_DELTA, 1, "const", True)


def stream_hw_cfg(d: StreamDesc, flags: int = 0) -> dict | None:
    """The v1 hardware's configuration for a descriptor (docs/stream.md, 4.3 / 4.4), or None if
    the board's engine cannot run it (the compiler then falls back to VOPs)."""
    if flags & (F_SRC_T | F_DST_T | F_NODST):
        return None
    if not (0 < d.rows <= DSTEP_MAX_ROWS) or d.cols % 64 or not (0 < d.cols <= 256):
        return None
    if d.q_self or d.out_p2 or d.o_reg_en or d.rinit or d.rsave or d.srs or d.drs:
        return None
    if d.u_mode != U_FMMA or d.b_src != B_SLOT or d.b_idx != 1 or d.d_reg != 2:
        return None
    g = {(G_CONST, 0): 0, (G_SLOT, 3): 1, (G_ONE, 0): 2}.get(
        (d.g_src, d.g_idx if d.g_src != G_ONE else 0))
    dm = DMODES.get(tuple(d.ops))
    if g is None or dm is None or (d.q_en and d.q_idx != 0):
        return None
    uses_a = dm in (0, 1, 3)
    if uses_a != d.a_en or (d.a_en and (d.a_op != A_SLOT or d.a_idx not in (1, 2))):
        return None
    return {"ns": d.cols // DSTEP_LANES, "rows": d.rows, "a_en": d.a_en,
            "a_sel": int(d.a_idx == 2), "dmode": dm, "g_src": g, "q_en": d.q_en}


def mm(sa, ssa, out, n, kb, rs, ors, m, ab, srs, unit=False, acc=False, rmax=False,
       ascale=None, wf=W8, pair=False, k0=0, rsc=False, ra=0, rb=0, rc=0, comment=""):
    """MM (docs/isa.md). With `ascale` (a TMEM address; needs unit and acc) the old accumulator
    is first multiplied by a per-row factor: y = T[out] * T[ascale + j] + a.w (the
    flash-attention rescale, done in the MXU epilogue). The address travels in the (unused)
    scale field. `wf`: the streamed weights' format (W8, W4I, W4F, WNV). `pair` (4-bit only):
    column reuse, ACT row j + m carries the odd K-blocks of row j ("Column reuse"). `k0`: the
    MM's first block within the scale rows at `ssa` (each row starts with its Sw). `rsc`: the
    scale rows are Sw alone (one word a row: the K cache)."""
    _check(0 < n < 65536 and 0 < kb < 256 and 0 < m < 256 and 0 <= ab < 256
           and 0 <= ors < 65536 and 0 <= k0 < 256,
           f"MM: n {n}, kb {kb}, m {m}, ab {ab}, ors {ors}, k0 {k0}")
    _check(wf in (W8, W4I, W4F, WNV), f"MM: weight format {wf}")
    _check(not pair or wf != W8, "MM: PAIR needs 4-bit weights")
    _check(not pair or k0 % 2 == 0, "MM: PAIR needs an even k0")
    _check(not rsc or wf != WNV, "MM: RSC does not take NVFP4")
    if ascale is not None:
        _check(unit and acc, "MM: ASCALE needs UNIT and ACC")
        ssa = ascale
    fl = (F_UNIT if unit else 0) | (F_ACC if acc else 0) | (F_RMAX if rmax else 0) | \
        (F_ASCALE if ascale is not None else 0) | (F_PAIR if pair else 0) | \
        (F_RSC if rsc else 0) | (wf << WF_SHIFT)
    return Instr(MM, ra=ra, rb=rb, rc=rc, flags=fl,
                 w=_w(sa, ssa, out, n | (kb << 16) | (k0 << 24), rs,
                      ors | (m << 16) | (ab << 24), srs),
                 comment=comment)


def tmm(sa, ssa, out, n, kb, rs, ors, m, ab, srs, unit=False, acc=False, rmax=False,
        ascale=None, rsc=False, pair=False, quad=False, ra=0, rb=0, rc=0, comment="",
        gsc=False, binary=False, grq=False, dense=False):
    """TMM: signed-int8 ACT dot packed ternary rows (00=0, 01=+1, 10=-1, 11=invalid).

    Four D-element K blocks share one D-byte weight chunk. Each non-unit weight row has one
    fp32 scale (RSC); its row stride is `srs`. The MXU applies the normal QACT activation
    scale/offset epilogue and the TMM's fixed 2^-11 weight exponent. `ascale` has MM's UNIT+ACC
    requirements. The packed operand starts at `sa`; K must begin on a four-block boundary when
    a compiler splits it into multiple TMM commands. `pair` (TMM PAIR, the K-split by two; a
    bitstream with Config.TPAIR): 2M <= MCOLS, the operand's rows also in ACT rows M..2M-1
    (QACT DUP); columns j take the even K blocks, j + M the odd ones, two blocks an advance
    (KB/2 cycles a weight row instead of KB), the same exact integer sums. `quad` (TMM QUAD,
    the K-split by four; a bitstream with Config.TQUAD): 4M <= MCOLS, the operand's rows also in
    ACT rows M..4M-1 (QACT DUP4); column j + q*M takes block 4c + q, a whole chunk an advance
    (ceil(KB/4) cycles a weight row), the same exact integer sums. Not with `pair`.
    `grq` (with `gsc`, ternary, a bitstream with Config.TGRQ): the block-major ("row quarter")
    layout. Block k's weights are a plane at sa + k*rs of N/4 D-byte chunks, chunk i holding rows
    4i..4i+3's block k (row 4i+q in quarter q); block k's scales a plane at ssa + k*srs of N fp32
    words (row n at + 4n). N % 4 == 0, M <= MCOLS, ssa and srs multiples of 16, srs >= 4N,
    rs >= N/4 * D. The same per-block fp32 math in K order as GSC, every chunk read once.
    """
    _check(0 < n < 65536 and 0 < kb < 256 and 0 < m < 256 and 0 <= ab < 256
           and 0 <= ors < 65536,
           f"TMM: n {n}, kb {kb}, m {m}, ab {ab}, ors {ors}")
    _check(0 <= rs < (1 << 32) and 0 <= srs < (1 << 32), f"TMM: rs {rs}, srs {srs}")
    _check(not rsc or not unit, "TMM: RSC is not used with UNIT")
    _check(unit or rsc or gsc, "TMM: non-unit weights require row scales (RSC)")
    if gsc:
        _check(not (unit or rsc or pair or quad or ascale is not None),
               "TMM GSC: UNIT, RSC, PAIR, QUAD and ASCALE are not supported")
        if grq:
            _check(not binary and n % 4 == 0 and srs % 16 == 0 and srs >= 4 * n and ssa % 16 == 0,
                   "TMM GRQ: ternary, N % 4 == 0, 16-byte scale planes of N words")
        else:
            _check(srs >= 4 * kb and srs % 4 == 0, "TMM GSC: scale row must hold KB fp32 words")
    _check(not grq or gsc, "TMM GRQ needs GSC")
    if ascale is not None:
        _check(unit and acc, "TMM: ASCALE needs UNIT and ACC")
        ssa = ascale
    fl = (F_UNIT if unit else 0) | (F_ACC if acc else 0) | (F_RMAX if rmax else 0) | \
        (F_ASCALE if ascale is not None else 0) | (F_RSC if rsc else 0) | (F_PAIR if pair else 0) | \
        (F_QUAD if quad else 0) | (F_GSC if gsc else 0)
    _check(not (pair and quad), "TMM: PAIR and QUAD are exclusive")
    _check(not binary or not (pair or quad),
           "TMM BIN: PAIR and QUAD are not supported")
    _check(not dense or (gsc and not (binary or grq or pair or quad)),
           "TMM T5 (dense trits): GSC only, not with BIN, GRQ, PAIR or QUAD")
    return Instr(TMM, ra=ra, rb=rb, rc=rc, flags=fl,
                 w=_w(sa, ssa, out, n | (kb << 16) | (TMM_BIN if binary else 0) | (TMM_GRQ if grq else 0) |
                      (TMM_T5 if dense else 0), rs,
                      ors | (m << 16) | (ab << 24), srs), comment=comment)


def tmm_cq(sa, ssa, out, n, kb, rs, ors, m, ab, srs, bits: int, acc=False, rmax=False,
           ra=0, rb=0, rc=0, comment="tmm cq"):
    """TMM CQ (Config.TCQ): GSC's per-(row, D-element block) fp32 scales with codebook weights:
    each code (bits 1: the BIN layout, eight blocks a chunk; 2: the ternary layout, four; 3 or
    4: 4-bit codes, two blocks a chunk, block 2c + h in half h, element i at nibble i of the
    half: byte i/2's low nibble for even i) selects the int8 LUT entry the last LUTSETs loaded
    (tmm_lut). The value of a weight is LUT[code] * Sw[n, k], as a ternary GSC weight's
    code * Sw[n, k]: Cactus CQ's norm[n, g] * codebook[code] with the codebook's int8 scale in
    Sw (opentpu/cq.py)."""
    _check(bits in (1, 2, 3, 4), f"TMM CQ: {bits}-bit codes")
    ins = tmm(sa, ssa, out, n, kb, rs, ors, m, ab, srs, acc=acc, rmax=rmax, ra=ra, rb=rb, rc=rc,
              comment=comment, gsc=True, binary=bits == 1)
    ins.w[3] |= TMM_CQ | (TMM_CQ4 if bits >= 3 else 0)
    return ins


def tmm_lut(lut, comment="lutset") -> list:
    """LUTSET: two TMMs loading the codebook LUT (16 int8 entries; fewer: zero-padded) that
    the next TMM CQs decode with. Each carries eight entries in its rs and srs words (entry i
    in byte i % 4 of rs for i < 4, of srs after), ors the half; N, KB, M 1, addresses 0. The
    MXU takes it in order, after every earlier TMM GSC (the LUT never changes under one)."""
    v = [int(x) for x in lut]
    _check(len(v) <= 16 and all(-128 <= x <= 127 for x in v), f"LUT {v}")
    b = bytes((x & 0xFF) for x in v + [0] * (16 - len(v)))
    out = []
    for h in range(2):
        w0 = int.from_bytes(b[8 * h:8 * h + 4], "little")
        w1 = int.from_bytes(b[8 * h + 4:8 * h + 8], "little")
        out.append(Instr(TMM, flags=F_GSC, w=_w(0, 0, 0, 1 | (1 << 16) | TMM_LUT, w0,
                                                h | (1 << 16), w1), comment=f"{comment} {h}"))
    return out


def qact(src, rows, ab, kb, srs, row=False, cscale=None, rscale=None, dup=False, dup4=False,
         ra=0, comment=""):
    """QACT. `cscale`/`rscale`: TMEM addresses of a per-column / per-row factor applied before
    quantization, x' = (x * T[rscale + r]) * T[cscale + c]. `dup`: ACT row r + rows receives
    a copy of row r in the same cycles (the odd K-blocks' column of an MM PAIR); `dup4` (with
    the TMM QUAD's bitstreams, Config.TQUAD): rows r + rows, r + 2 rows and r + 3 rows do
    (4 rows <= MCOLS)."""
    _check(0 < rows < 256 and 0 <= ab < 256 and 0 < kb < 65536,
           f"QACT: rows {rows}, ab {ab}, kb {kb}")
    fl = (F_ROW if row else 0) | (F_CSCALE if cscale is not None else 0) | \
        (F_RSCALE if rscale is not None else 0) | (F_DUP if dup or dup4 else 0) | \
        (F_DUP4 if dup4 else 0)
    return Instr(QACT, ra=ra, flags=fl,
                 w=_w(src, rows | (ab << 8) | (kb << 16), srs, cscale or 0, rscale or 0),
                 comment=comment)


def qst(src, dst, sdst, rows, kb, srs, drs, es, row=False, half=False, q4=False, ra=0, rb=0,
        rc=0, comment=""):
    """QST; q4 (Config.QST4): ROW mode, es 1, every row's elements as int4 nibbles, two a byte
    (element c in byte dst + r*drs + c/2, the low nibble for even c), S = amax * f32(1/7) and
    q = clamp(rne(x * (7 * recip(amax))), -7, 7): an int4 MM operand with RSC (a K cache)."""
    _check(0 < rows < 65536 and 0 < kb < 65536, f"QST: rows {rows}, kb {kb}")
    _check(row or not half, "QST: HALF needs ROW mode")
    _check(not q4 or (row and es == 1 and not half), "QST Q4: ROW mode, es 1, not HALF")
    return Instr(QST, ra=ra, rb=rb, rc=rc, flags=(F_ROW if row else 0) | (F_HALF if half else 0)
                 | (F_Q4 if q4 else 0),
                 w=_w(src, dst, sdst, rows | (kb << 16), srs, drs, es), comment=comment)


def vop(func, dst, a, b, rows, cols, drs, ars, brs, bmode=B_FULL, imm=0.0,
        ra=0, rb=0, rc=0, rd=0, comment=""):
    """imm: an fp32 immediate (ARGMAX: an int, the index base)."""
    _check(0 < rows < 65536 and 0 < cols < 65536, f"VOP: rows {rows}, cols {cols}")
    _check(0 <= drs < 65536 and 0 <= ars < 65536 and 0 <= brs < 65536,
           f"VOP: drs {drs}, ars {ars}, brs {brs} (16 bits)")
    iw = u32(imm) if func in (V_ARGMAX, V_MAC, V_SEL, V_RAND, V_BFLY, V_AFF) else f32bits(imm)
    return Instr(VOP, ra=ra, rb=rb, rc=rc, rd=rd,
                 w=_w(dst, a, b, rows | (cols << 16), drs | (ars << 16),
                      brs | (func << 16) | (bmode << 24), iw),
                 comment=comment)


def rand(dst, a, rows, cols, drs, ars, key=0, ra=0, rb=0, rd=0, comment="rand"):
    """VOP RAND (Config.VRAND): T[dst + r*drs + j] = uniform(bits(T[a + r*ars + j]), key + R[rd])
    for the counter-based generator of opentpu/rand.py: an fp32 in [0, 1 - 2^-24], a function of
    the counter word (A's raw bits, e.g. an iota table's) and the key (raw u32) only."""
    return vop(V_RAND, dst, a, 0, rows, cols, drs, ars, 0, B_FULL, key & 0xFFFFFFFF,
               ra=ra, rb=rb, rd=rd, comment=comment)


def bfly(dst, a, rows, cols, drs, ars, s, ra=0, rb=0, comment="bfly"):
    """VOP BFLY (Config.VBFLY): one butterfly stage of the Hadamard transform at stride s over
    each row of A [rows, cols]: dst(r, c) = A(r, c) + A(r, c ^ s) for c & s == 0, else
    A(r, c ^ s) - A(r, c). s a power of two, cols % 2s == 0; dst must not overlap A."""
    _check(s > 0 and s & (s - 1) == 0 and cols % (2 * s) == 0,
           f"BFLY: stride {s}, cols {cols}: a power of two with cols % 2s == 0")
    return vop(V_BFLY, dst, a, a, rows, cols, drs, ars, ars, B_FULL, s,
               ra=ra, rb=rb, rc=rb, comment=comment)


def aff(dst, a, p, gh, rows, cols, drs, ars, prs=1, bias=True, ra=0, rb=0, rc=0, rd=0,
        comment="aff"):
    """VOP AFF (Config.VAFF): dst(r, c) = (A(r, c) * T[p + r*prs]) * G(c) + H(c), G(c) =
    T[R[rd] + gh + c] and H(c) = T[R[rd] + gh + cols + c] (bias False: H = -0, flag DONE);
    each step a rounded mul then a rounded add, as MUL then MAC. cols <= 256; G and H are read
    into the column buffers before anything is written."""
    _check(0 < cols <= OUTER_MAX_COLS, f"AFF: cols {cols} (at most {OUTER_MAX_COLS})")
    ins = vop(V_AFF, dst, a, p, rows, cols, drs, ars, prs, B_ROW, gh, ra=ra, rb=rb, rc=rc,
              rd=rd, comment=comment)
    ins.flags = 0 if bias else F_DONE
    return ins


def argmax(dst, a, rows, cols, drs=2, ars=0, base=0, ra=0, rb=0, rd=0, comment="argmax"):
    """VOP ARGMAX: T[dst + r*drs] = the row maximum (RMAX), T[dst + r*drs + 1] =
    i2f(c + R[rd] + base) for c the first column holding it (rows > 1: drs >= 2)."""
    _check(rows == 1 or drs >= 2, "ARGMAX: the rows' (max, index) pairs need drs >= 2")
    return vop(V_ARGMAX, dst, a, 0, rows, cols, drs, ars, 0, B_FULL, base, ra=ra, rb=rb, rd=rd,
               comment=comment)


def outer(dst, d, b, c, rows, cols, drs, brs, dmode="scalar", ra=0, rb=0, rc=0, rd=0,
          comment=""):
    """VOP OUTER: T[dst + r*drs + j] = T[dst + r*drs + j] * Dv(j) + T[b + r*brs] * T[c + j],
    Dv(j) = T[d] (dmode "scalar"), T[d + j] ("column") or 1.0 ("one", d unused). The decay
    address travels in the A field and the column vector's in the immediate word (w7 += R[rd])."""
    _check(0 < rows < 65536 and 0 < cols <= OUTER_MAX_COLS and 0 <= drs < 65536
           and 0 <= brs < 65536, f"OUTER: rows {rows}, cols {cols}, drs {drs}, brs {brs}")
    fl = {"scalar": F_DSCALAR, "column": 0, "one": F_DONE}[dmode]
    return Instr(VOP, ra=ra, rb=rb, rc=rc, rd=rd, flags=fl,
                 w=_w(dst, d, b, rows | (cols << 16), drs, brs | (V_OUTER << 16) | (B_ROW << 24),
                      c),
                 comment=comment)


def mac(dst, a, b, c, rows, cols, drs, ars, brs, bmode=B_FULL, ra=0, rb=0, rc=0, rd=0,
        comment="mac"):
    """VOP MAC: T[dst + r*drs + j] = A(r, j) * T[c + j] + B(r, j) (B in bmode FULL, ROW or COL;
    the column vector Cv = T[c + j] is read first, into the VPU's column buffers: cols <= 256)."""
    _check(0 < cols <= OUTER_MAX_COLS and bmode != B_SCALAR,
           f"MAC: cols {cols} (at most {OUTER_MAX_COLS}), bmode {bmode} (not SCALAR)")
    return vop(V_MAC, dst, a, b, rows, cols, drs, ars, brs, bmode, c, ra=ra, rb=rb, rc=rc,
               rd=rd, comment=comment)


def sel(dst, a, b, mask, rows, cols, drs, ars, brs, bmode=B_FULL,
        ra=0, rb=0, rc=0, rd=0, comment="sel"):
    """Select A where the buffered column mask is nonzero, otherwise B.

    The mask is read before any output, and applies to every row. For independent
    row masks issue one instruction per row. Values are flushed to zero as usual.
    """
    _check(0 < cols <= OUTER_MAX_COLS and bmode != B_SCALAR,
           f"SEL: cols {cols} (at most {OUTER_MAX_COLS}), bmode {bmode} (not SCALAR)")
    return vop(V_SEL, dst, a, b, rows, cols, drs, ars, brs, bmode, mask,
               ra=ra, rb=rb, rc=rc, rd=rd, comment=comment)


def gather(src, dst, rows, cols, srs, drs, seg, ra=0, rb=0, comment=""):
    _check(0 < rows < 65536 and 0 < cols < 65536, f"GATHER: rows {rows}, cols {cols}")
    return Instr(GATHER, ra=ra, rb=rb, w=_w(src, dst, rows | (cols << 16), srs, drs, seg),
                 comment=comment)


def assemble(prog: list[Instr]) -> np.ndarray:
    return np.array([x for ins in prog for x in ins.encode()], dtype=np.uint32)


def disassemble(prog: list[Instr]) -> str:
    return "\n".join(f"{i:4d}: {ins}" for i, ins in enumerate(prog))
