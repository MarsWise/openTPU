"""r6 compare and column-mask select, independently checked and RTL matched to ISA."""
import numpy as np
import pytest

from opentpu import Config, isa as I, rtlsim
from opentpu.isasim import Machine, SimError
from opentpu.target import arch_hash
import opentpu.language as ol
from opentpu.runtime import Input, Output, compile_kernel, run


def program():
    return [I.ld(0, 0, 512),
            I.vop(I.V_CMP_GT, 1024, 0, 128, 2, 32, 32, 32, 32),
            I.vop(I.V_CMP_EQ, 1152, 0, 128, 2, 32, 32, 32, 32),
            I.sel(1280, 0, 128, 1024, 2, 32, 32, 32, 32),
            I.st(0x8000, 1024, 320), I.halt()]


def image():
    rng = np.random.default_rng(831)
    src = rng.normal(size=512).astype(np.float32)
    edges = np.array([0x00000000, 0x80000000, 0x00000001, 0x80000001,
                      0x7f800000, 0xff800000, 0x7fc00001, 0xffc00001], np.uint32)
    src[:8] = edges.view(np.float32)
    src[128:136] = edges[::-1].view(np.float32)
    src[32:40] = edges.view(np.float32)
    src[160:168] = edges.view(np.float32)
    dram = np.zeros(1 << 20, dtype=np.uint8)
    dram[:src.nbytes] = src.view(np.uint8)
    return src, dram


def flushed(x):
    bits = x.view(np.uint32).copy()
    bits[(bits & 0x7f800000) == 0] &= np.uint32(0x80000000)
    return bits.view(np.float32)


def test_comparisons_and_selection_against_numpy():
    src, dram = image()
    sim = Machine(Config(VCMP=True), [program()], [dram]).run().slices[0]
    a, b = flushed(src[:64].reshape(2, 32)), flushed(src[128:192].reshape(2, 32))
    with np.errstate(invalid="ignore"):
        gt, eq = (a > b).astype(np.float32), (a == b).astype(np.float32)
    np.testing.assert_array_equal(sim.tmem[1024:1088].view(np.float32), gt.reshape(-1))
    np.testing.assert_array_equal(sim.tmem[1152:1216].view(np.float32), eq.reshape(-1))
    expected = np.where(gt[0][None, :] != 0, a, b)
    np.testing.assert_array_equal(sim.tmem[1280:1344].view(np.uint32), expected.reshape(-1).view(np.uint32))


def test_old_firmware_and_config_are_distinct():
    _, dram = image()
    with pytest.raises(SimError, match="Config.VCMP"):
        Machine(Config(), [program()], [dram]).run()
    assert arch_hash(Config()) != arch_hash(Config(VCMP=True))


def test_cmp_sel_rtl(have_verilator):
    _, dram = image()
    cfg = Config(VCMP=True)
    expected = Machine(cfg, [program()], [dram.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [program()], [dram.copy()])
    np.testing.assert_array_equal(tmems[0], expected.slices[0].tmem)
    np.testing.assert_array_equal(drams[0], expected.slices[0].dram)


@ol.jit
def _select_kernel(a, b, mask, y, M: int, N: int):
    x, z, m = ol.load(a), ol.load(b), ol.load(mask)
    ol.store(y, ol.select_columns(m, x, z))


def test_compiler_select_splits_columns_and_preserves_mask():
    rng = np.random.default_rng(71)
    a = rng.standard_normal((3, 300)).astype(np.float32)
    b = rng.standard_normal((3, 300)).astype(np.float32)
    mask = rng.integers(0, 2, size=300).astype(np.float32)
    args = dict(a=Input(a), b=Input(b), mask=Input(mask), y=Output(a.shape), M=3, N=300)
    comp, images = compile_kernel(_select_kernel, Config(VCMP=True), **args)
    got = run(comp, images, "isa").outputs["y"]
    np.testing.assert_array_equal(got, np.where(mask[None, :] != 0, a, b))


def test_board_model_advertises_capabilities(have_verilator):
    from opentpu.host import regs
    from opentpu.host.board import SimTransport
    t = SimTransport(ch_bytes=1 << 20, params={"VCMP": 1, "TBIN": 1, "TGSC": 1,
                                               "TBGSC": 1, "VU8": 1, "COLL_GATHER": 0})
    c = regs.caps2(t.reg_read(regs.R_CAPS2))
    assert c["vcmp"] and c["tbin"] and c["tgsc"] and c["tbgsc"] and c["vu8"]
