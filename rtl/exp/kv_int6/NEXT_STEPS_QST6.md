# QST6 and INT6 KV: proposed next integration work

Status: design investigation and isolated experimental RTL only. No complete
QST6 ISA or integrated QST6 RTL, MXU format or physical inference has been
implemented or approved. The local codec format is a candidate, not an existing
openTPU or checkpoint contract.

Update: `otpu_qst6_row_stream` now demonstrates streamed groups with one
externally supplied maximum and one scale per row; `otpu_kv_int6_unpack16`
provides signed-byte expansion. See `ROW_STREAM_STATUS.md` for exact test and
OOC evidence. Neither block is integrated with the existing units. In
particular the per-16-group automatic scale experiment is not interchangeable
with a full-row W8/RSC scale, and the new common-scale writer does not compute
the full-row maximum or establish an MXU read path.

## Existing behavior

* `otpu_quant.sv` QST supports row scales, optional HALF, and Q4 on flag bit 2.
  `otpu_qscale` implements factors 7/127. Q4 uses eight elements per 32-bit
  output word; legacy byte writes use byte enables. Its reads, writer metadata,
  arbitration grant and `wr_idle` acknowledgements are pipelined together.
* `otpu_pkg.sv` footprint for QST derives the data range from `KB*D`, row
  stride and element stride, then rounds to 64-byte beats. `otpu_seq.sv`
  scores data and scale ranges. Preserve this safety for every command; extra
  stalls due to a conservative overestimate are acceptable, missing conflicts
  are not. Address wrap and end-of-4-GiB behavior need tests.
* `otpu_mxu.sv` has no INT6 weight/KV format. Its two-bit MM weight-format field
  is already used by W8/W4I/W4F/WNV. A decoder cannot be added by assuming an
  unused encoding without a separate compatibility and capability contract.

## Minimal staged path

1. Specify the numerical quantizer separately from the codec. A candidate is
   per-row `S = amax * f32(1/31)`, `inv = 31 * recip(amax)`, round-to-nearest
   even, signed saturation `[-31,31]`; codec itself represents `[-32,31]`.
   This is a proposed choice, not verified exact hardware behavior. Define
   zero, FTZ/subnormal, NaN, infinity, rounding and clipping, and validate
   against the existing fp32 semantics before modifying RTL.
2. Reserve a distinct QST6 feature/capability and opcode flag. QST flag bit 3
   and CAPS2 bit 18 are candidates only: verify across all instruction decoders,
   targets, build hashes, host version checks and other branches first. Legacy
   instructions must retain their bit-exact behavior when the feature is off.
3. First support **contiguous K rows only**, not V-transpose: ROW, ES=1,
   HALF=0, no T4 path, D=128, `dst`/`drs` word-aligned and full groups of
   16 elements. A row has `3*KB*D/4` bytes; one fp32 scale per row. Enforce
   actual buffer size, scale/data non-overlap and 32-bit address limits before
   dispatch. No claims about arbitrary tails or sub-byte writes.
4. Four 6-bit codes form 3 bytes, so successive groups are not word-aligned.
   A safe implementation needs to collect 16 values (96 packed bits) and emit
   three 32-bit words, or an equivalent proven byte-carry scheme. With LANES=8,
   input quantization and writes have different rates: explicitly buffer and
   backpressure, keep valid/data/address/byte-enables together across `gnt`
   stalls, and only finish after all writes and `wr_idle` acknowledgement.
   A simple one-word write per four values will overwrite/drop cross-word bytes.
5. Extend hazard ranges conservatively and prove them against exact bytes for
   every accepted Q6 encoding and row/stride. Tests must cover read-after-write,
   write-after-read and write-after-write against DMA, MXU, QST and scale writes,
   WAITC on/off, backpressure, last byte of 64-byte beat, 4 KiB and 4 GiB edges.
6. Define and implement an INT6 MXU read format, alignment, scales, decoder,
   ISA/config/host guards and simulator. Test against an independent int6 decode
   and attention reference. OOC and full synthesis, route and hardware results
   are separate gates.

## Why V-transpose is not done by contiguous QST6

The existing V cache stores a transposed token column. Adjacent 6-bit tokens
share bytes; a single token update may affect bits of two bytes. A byte-enable
can preserve bytes, not individual bits. Full groups, a correctly serialized
read-modify-write path, or a different device-resident layout/repacking scheme
are required. Either design needs conflict accounting and verification for
neighbouring token updates, stalls and reset. Do not mark V INT6 supported
because the four-value codec or contiguous K writes work.

Before firmware integration: complete the mandatory `rtl-review-signoff` hook
with `bin/rtl-lint`/slang, full reviews, protocol/hazard tests, source-identified
synthesis, timing and quality measurements. The hook is unavailable in the
current environment; `REVIEW_STATUS.md` records the experimental codec's status.
