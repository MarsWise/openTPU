"""WAITC must conservatively match production RTL on 32-bit wrap."""
from opentpu import isa as I, waits as W
import random


def test_waitc_scores_wrapped_qst_against_low_dram_store():
    qst = I.qst(0, 0xfffffff0, 0x800, 1, 1, 32, 32, 1, row=True)
    store = I.st(0, 0x200, 4)
    entries = W.wait_counts([qst, store, I.halt()], D=32, args=[0] * 8)
    assert W.counts(entries[1])[W.U_Q] == 1


def test_waitc_exact_top_end_is_conservatively_covered():
    qst = I.qst(0, 0xffffffe0, 0x800, 1, 1, 32, 32, 1, row=True)
    store = I.st(0, 0x200, 4)
    entries = W.wait_counts([qst, store, I.halt()], D=32, args=[0] * 8)
    assert W.counts(entries[1])[W.U_Q] == 1


def test_waitc_scores_qst_span_addition_overflow():
    qst = I.qst(0, 0x100, 0x800, 2, 1, 32, 0xfffffff0, 1, row=True)
    store = I.st(0xf0, 0x200, 4)
    entries = W.wait_counts([qst, store, I.halt()], D=32, args=[0] * 8)
    assert W.counts(entries[1])[W.U_Q] == 1


def test_waitc_scores_qst_stride_product_overflow():
    qst = I.qst(0, 0x100, 0x800, 3, 1, 32, 0x80000000, 1, row=True)
    store = I.st(0x80000100, 0x200, 4)
    entries = W.wait_counts([qst, store, I.halt()], D=32, args=[0] * 8)
    assert W.counts(entries[1])[W.U_Q] == 1


def test_waitc_scores_qst_element_stride_product_overflow():
    qst = I.qst(0, 0x100, 0x800, 1, 1, 32, 32, 0x9f767c46, row=True)
    store = I.st(0x9f767d44, 0x200, 1)
    entries = W.wait_counts([qst, store, I.halt()], D=32, args=[0] * 8)
    assert W.counts(entries[1])[W.U_Q] == 1


def test_waitc_scores_qst_source_product_overflow():
    qst = I.qst(0, 0x100, 0x800, 3, 1, 0x80000000, 32, 1, row=True)
    load = I.ld(0x50000000, 0x80000000, 1)
    entries = W.wait_counts([qst, load, I.halt()], D=32, args=[0] * 8)
    assert W.counts(entries[1])[W.U_Q] == 1


def test_waitc_scores_qst_scale_span_overflow():
    qst = I.qst(0, 0x10000000, 0x1000, 65535, 65535, 0, 32, 1)
    store = I.st(0, 0x80000000, 1)
    entries = W.wait_counts([qst, store, I.halt()], D=32, args=[0] * 8)
    assert W.counts(entries[1])[W.U_Q] == 1


def test_waitc_scores_dma_length_overflow_against_qst():
    qst = I.qst(0, 0x10000, 0x20000, 1, 1, 32, 32, 1, row=True)
    # DMA's word count becomes 2^32 bytes; it must not become an empty range.
    for dma in (I.st(0x40000000, 0x10000, 1 << 30), I.ld(0x40000000, 0x10000, 1 << 30)):
        entries = W.wait_counts([qst, dma, I.halt()], D=32, args=[0] * 8)
        assert W.counts(entries[1])[W.U_Q] == 1


def test_waitc_scores_ldx_table_product_overflow():
    store = I.st(0x3fc10000, 0x10000, 1)
    gather = I.ldx(0x10000, 0x400, 0x800, 1, 1, 0x400000, 1, 1026, D=128)
    # Both instructions use the DMA and are issued in order; check the raw
    # footprint directly rather than expecting a same-unit wait entry.
    regs = [(0, 0)] * 16
    assert W.conflict(W.footprint(store, regs, 128), W.footprint(gather, regs, 128))


def test_waitc_scores_qact_source_product_overflow():
    qact = I.qact(0x400, 3, 0, 1, 0x80000000)
    load = I.ld(0x50000000, 0x80000400, 1)
    entries = W.wait_counts([qact, load, I.halt()], D=128, args=[0] * 8)
    assert W.counts(entries[1])[W.U_Q] == 1


def test_waitc_scores_ldx_destination_product_overflow():
    gather = I.ldx(0x10000, 0x400, 0x800, 3, 1, 128, 0x80000000, 1, D=128)
    load = I.ld(0x50000000, 0x80000400, 1)
    regs = [(0, 0)] * 16
    assert W.conflict(W.footprint(gather, regs, 128), W.footprint(load, regs, 128))


def test_waitc_scores_mm_weight_stride_product_overflow():
    mm = I.mm(0x10000, 0, 0x400, 3, 1, 0x80000000, 3, 1, 0, 0, unit=True)
    store = I.st(0x80010000, 0x200, 1)
    entries = W.wait_counts([mm, store, I.halt()], D=128, args=[0] * 8)
    assert W.counts(entries[1])[W.U_MXU] == 1


def test_waitc_scores_gsc_scale_stride_product_overflow():
    tmm = I.tmm(0x10000, 0x30000, 0x400, 3, 1, 128, 3, 1, 0, 0x80000000, gsc=True)
    store = I.st(0x80030000, 0x200, 1)
    entries = W.wait_counts([tmm, store, I.halt()], D=128, args=[0] * 8)
    assert W.counts(entries[1])[W.U_MXU] == 1


def test_waitc_scores_mm_weight_and_scale_span_addition_overflow():
    weight_wrap = I.mm(0x10000, 0, 0x400, 2, 1, 0xffffff80, 2, 1, 0, 0, unit=True)
    scale_wrap = I.mm(0x10000, 0x30000, 0x400, 2, 1, 128, 2, 1, 0, 0xfffffffc, rsc=True)
    for mm, addr in ((weight_wrap, 0xff80), (scale_wrap, 0x2fffc)):
        store = I.st(addr, 0x200, 1)
        entries = W.wait_counts([mm, store, I.halt()], D=128, args=[0] * 8)
        assert W.counts(entries[1])[W.U_MXU] == 1


def test_waitc_scores_gather_source_stride_overflow():
    gather = I.gather(0x400, 0x1000, 3, 1, 0x80000000, 1, 1)
    load = I.ld(0x50000000, 0x80000400, 1)
    entries = W.wait_counts([gather, load, I.halt()], D=128, S=1, args=[0] * 8)
    assert W.counts(entries[1])[W.U_COLL] == 1


def test_waitc_scores_gather_destination_product_and_sum_overflow():
    products = I.gather(0x1000, 0x400, 3, 1, 1, 0x80000000, 1)
    addition = I.gather(0x1000, 0x100, 2, 32, 32, 0xfffffff0, 1)
    segments = I.gather(0x1000, 0x400, 1, 1, 1, 1, 0x80000000)
    for gather, addr, slices in ((products, 0x80000400, 1), (addition, 0xf0, 1),
                                 (segments, 0x80000400, 3)):
        store = I.st(0x50000000, addr, 1)
        entries = W.wait_counts([gather, store, I.halt()], D=128, S=slices, args=[0] * 8)
        assert W.counts(entries[1])[W.U_COLL] == 1


def test_waitc_scores_dstep_scale_offset_overflow():
    dstep = I.dstep(0x10000, 0x200, 0x300, 1, 64, 0x400, 0xffffffff, 0x500)
    overwrite = I.vop(I.V_COPY, 0x3ff, 0x900, 0, 1, 1, 1, 1, 1)
    entries = W.wait_counts([dstep, overwrite, I.halt()], D=128, args=[0] * 8)
    assert W.counts(entries[1])[W.U_DMA] == 1


def test_waitc_register_range_containing_a_wrapping_base_is_conservative():
    assert W._mk(W.SP_DRAM, (0xfffffff0, 0xfffffff8), 16) == [(W.SP_DRAM, 0, W.M32)]
    # Base values themselves may wrap; if any resulting RTL range wraps, its
    # union includes the entire conservative interval.
    assert W._mk(W.SP_DRAM, (0xfffffff0, 0x100000003), 4) == [(W.SP_DRAM, 0, W.M32)]
    assert W._mk(W.SP_DRAM, (0x100, 0x110), 16) == [(W.SP_DRAM, 0x100, 0x120)]


def test_waitc_union_covers_sampled_rtl_ranges():
    rng = random.Random(0x4A17)
    for _ in range(2000):
        lo = rng.randrange(1 << 32)
        if rng.randrange(2):
            lo = 0xfffff000 + rng.randrange(4096)
        hi = lo + rng.randrange(257)
        length = rng.choice([0, 1, 2, 4, 16, 64, rng.randrange(1, 1024)])
        intervals = W._mk(W.SP_DRAM, (lo, hi), length)
        for value in {lo, hi, (lo + hi) // 2, min(hi, 0xffffffff)}:
            if value < lo or value > hi or not length:
                continue
            base = value & W.M32
            end = base + length
            rtl_lo = 0 if end >= 1 << 32 else base
            rtl_hi = min(end, W.M32)
            if rtl_hi <= rtl_lo:
                continue
            assert any(low <= rtl_lo and upper >= rtl_hi for _, low, upper in intervals), (
                lo, hi, length, base, intervals, rtl_lo, rtl_hi)
