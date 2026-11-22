"""The MXU's optional weight formats as build parameters (otpu_mxu HAS_BIN / HAS_CQ / HAS_T5;
otpu_slice HAS_TBIN / HAS_TCQ / HAS_TD5, from the board's TBIN / TCQ / TD5): a build without
them leaves their decode out (area) and still computes every other format bit for bit."""
import re
from pathlib import Path

import numpy as np
import pytest

from opentpu import isa as I
from opentpu.isasim import Config, Machine

ROOT = Path(__file__).resolve().parents[1]
OFF = {"HAS_TBIN": 0, "HAS_TCQ": 0, "HAS_TD5": 0, "HAS_TATAN": 0}


def test_the_formats_are_parameters_down_to_the_mxu():
    mxu = (ROOT / "rtl/mxu/otpu_mxu.sv").read_text()
    for p in ("HAS_BIN", "HAS_CQ", "HAS_T5"):
        assert re.search(rf"parameter bit {p}\s*=\s*1'b1", mxu), p
    # each format's command decode and every use after the FIFO are gated
    assert "wire cmd_t5 = HAS_T5 &&" in mxu and "wire cmd_binary = HAS_BIN &&" in mxu
    assert "wire cmd_cq = HAS_CQ &&" in mxu
    assert "c_binary = HAS_BIN && pop_e.binary" in mxu
    assert not re.search(r"pop_e\.(t5|cq4?)\b", mxu.split("c_t5 = HAS_T5 && pop_e.t5;")[1])
    assert "if (HAS_CQ) begin : g_cqh" in mxu                 # the D LUT RAMs
    slice_ = (ROOT / "rtl/top/otpu_slice.sv").read_text()
    assert ".HAS_BIN(HAS_TBIN), .HAS_CQ(HAS_TCQ), .HAS_T5(HAS_TD5)) u_mxu" in slice_
    board = (ROOT / "rtl/shell/otpu_board.sv").read_text()
    assert board.count(".HAS_TBIN(TBIN), .HAS_TCQ(TCQ), .HAS_TD5(TD5), .HAS_TATAN(TATAN)) u_slice") == 2
    top = (ROOT / "rtl/top/otpu_top.sv").read_text()
    assert ".HAS_TBIN(HAS_TBIN), .HAS_TCQ(HAS_TCQ), .HAS_TD5(HAS_TD5), .HAS_TATAN(HAS_TATAN))" in top
    tb = (ROOT / "sim/verilator/tb_top.sv").read_text()
    assert ".HAS_TBIN(HAS_TBIN), .HAS_TCQ(HAS_TCQ), .HAS_TD5(HAS_TD5), .HAS_TATAN(HAS_TATAN)) dut" in tb


def _mm_program(cfg):
    from tests.test_rtl import DATA, INT8, MX_BASE, MX_PITCH, _images
    D, KB, N = cfg.D, 3, 20
    prog = [I.ld(DATA, 0, 4096)]
    out = 4096
    for M, wf, pair in ((3, I.W8, False), (2, I.W4F, True), (2, I.WNV, True), (5, I.W4I, False)):
        rs = KB * D if wf == I.W8 else -(-KB // 2) * D
        prog += [I.qact(0, M, 0, KB, KB * D, dup=pair),
                 I.mm(INT8, MX_BASE[wf], out, N, KB, rs, N, M, 0, MX_PITCH[wf], wf=wf,
                      pair=pair)]
        out += 4 * N * M
    return prog + [I.halt()], _images(np.random.default_rng(11), 1)[0]


@pytest.mark.parametrize("uarch", [OFF, {"HAS_TCQ": 0}])
def test_mm_formats_without_the_optional_decodes_rtl(have_verilator, uarch):
    """int8, int4, fp4 (PAIR) and NVFP4 MMs: RTL without BIN / CQ / T5 == ISA."""
    from opentpu import rtlsim
    cfg = Config(S=1, MCOLS=8, ACT_ROWS=16)
    prog, img = _mm_program(cfg)
    ref = Machine(cfg, [prog], [img.copy()]).run().slices[0]
    drams, tmems, _ = rtlsim.run(cfg, [prog], [img.copy()], uarch=uarch)
    np.testing.assert_array_equal(tmems[0], ref.tmem)
    np.testing.assert_array_equal(drams[0], ref.dram)


@pytest.mark.parametrize("m,acc", [(5, True), (8, False)])
def test_ternary_gsc_grq_quad_without_the_optional_decodes_rtl(have_verilator, m, acc):
    """Ternary TMM GSC, GRQ and GRQ QUAD (Bonsai's, BitNet's 2-bit codes) on the m8 MXU built
    without BIN / CQ / T5: RTL == ISA under AXI stalls."""
    from opentpu import rtlsim
    from tests.test_tmm_grq import case, gq4_case
    cfg, image, head, gsc, grq, _ = case(6, m, acc, 16, mcols=8)
    prog = head + [grq, gsc, I.halt()]
    ref = Machine(cfg, [prog], [image.copy()]).run().slices[0]
    drams, tmems, _ = rtlsim.run(cfg, [prog], [image.copy()], axi=True, stall=35, seed=m,
                                 uarch=OFF)
    np.testing.assert_array_equal(tmems[0], ref.tmem)
    np.testing.assert_array_equal(drams[0], ref.dram)
    cfg, image, head, grq, quad = gq4_case(5, acc, 12, mcols=8)
    prog = head + [quad, grq, quad, I.halt()]
    ref = Machine(cfg, [prog], [image.copy()]).run().slices[0]
    drams, tmems, _ = rtlsim.run(cfg, [prog], [image.copy()], axi=True, stall=35, seed=m + 1,
                                 uarch=OFF)
    np.testing.assert_array_equal(tmems[0], ref.tmem)
    np.testing.assert_array_equal(drams[0], ref.dram)


def test_the_build_leaves_capabilities_out_and_hashes_what_it_builds():
    ps = (ROOT / "tools/build_r5_timing.ps1").read_text()
    assert "$env:OTPU_DROP_FEATURES" in ps and "$buildEnv[$k] = '0'" in ps
    assert "TBIN needs TBGSC left out too" in ps
    # CFG_HASH from $buildEnv (what is built), not from the switches
    assert "('{0}=bool({1})' -f $c, [int]($buildEnv[\"OTPU_${c}_GEN\"] -eq '1'))" in ps
    assert "TGRQ=bool($gsc)" not in ps
    cmd = (ROOT / "tools/nextgen/card_runs/r8/r8_build_c4.cmd").read_bytes().decode()
    # TCQ stays: the 6-bit KV cache (kv6 / k6v6: QST6 + TMM CQ2 GRQ), Needle CQ4
    assert 'set "OTPU_DROP_FEATURES=TD5 TBIN TBGSC"' in cmd and "\r\n" in cmd
    assert "hash=37f51889" in cmd
    assert "set OTPU_LITEDRAM_RTW=4" in cmd and "set TRACE_DEPTH=4096" in cmd
    # each dropped capability is one the board's ctrl reports and create_project.tcl sets
    tcl = (ROOT / "boards/ypcb-00338/vivado/create_project.tcl").read_text()
    for f in ("TD5", "TBIN", "TBGSC"):
        assert f"OTPU_{f}_GEN" in ps and f"OTPU_{f}_GEN" in tcl


def test_the_host_refuses_the_left_out_formats():
    """A bitstream without TBIN / TCQ / TD5 says so in CAPS2; Board.load_program refuses their
    TMMs (ConfigMismatch) rather than running them on a decode that is not there."""
    src = (ROOT / "opentpu/host/board.py").read_text()
    for cap in ('"tbin"', '"tcq"', '"td5"'):
        assert f'.get("caps2", {{}}).get({cap}, False)' in src, cap


def test_the_c4_hash_is_the_capabilities_built():
    import dataclasses
    from opentpu.target import arch_hash
    from tools.bonsai2_validate import configuration
    on = dict(TGSC=1, VCMP=1, TBIN=1, TBGSC=1, VU8=1, TGRQ=1, VRAND=1, TCQ=1, TATAN=1, QST4=1,
              LDX=1, TD5=1, VBFLY=1, VAFF=1, QST6=1, VCSWP=1, TGQ4=1)

    def h(d):
        return arch_hash(dataclasses.replace(configuration(), **{k: bool(v) for k, v in d.items()}))
    assert h(on) == 0xa905c3db                                  # C2 / C3
    assert h(dict(on, TD5=0, TBIN=0, TBGSC=0)) == 0x37f51889    # C4 (r8_build_c4.cmd)
    assert h(dict(on, TBIN=0, TBGSC=0)) == 0x432e34c2           # A (r8_build_a.cmd)


def test_r8_card_check_takes_the_area_build():
    import importlib.util
    spec = importlib.util.spec_from_file_location("r8cc", ROOT / "tools/r8_card_check.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    full = {k: True for k in m.R8_CAPS2}
    full["tgq4"] = True
    c3 = {"caps2": full, "cfg": {"hash": 0xa905c3db}}
    assert all(ok for ok, _, _ in m.caps_checks(c3))
    c4 = {"caps2": {**full, "tbin": False, "tbgsc": False, "td5": False},
          "cfg": {"hash": 0x37f51889}}
    assert all(ok for ok, _, _ in m.caps_checks(c4))
    # a C3 hash without T5 is wrong, and so is an area build that still reports T5
    bad = {"caps2": {**full, "td5": False}, "cfg": {"hash": 0xa905c3db}}
    assert not all(ok for ok, _, _ in m.caps_checks(bad))
    bad = {"caps2": {**full, "tbin": False, "tbgsc": False}, "cfg": {"hash": 0x37f51889}}
    assert not all(ok for ok, _, _ in m.caps_checks(bad))
    # experiment A: less BIN / BIN GSC, T5 kept
    a = {"caps2": {**full, "tbin": False, "tbgsc": False}, "cfg": {"hash": 0x432e34c2}}
    assert all(ok for ok, _, _ in m.caps_checks(a))


@pytest.mark.parametrize("KB,unit,acc,M", [(7, False, True, 2), (20, False, False, 1),
                                           (5, True, False, 2)])
def test_bitnet_tmm_plain_pair_quad_without_the_optional_decodes_rtl(have_verilator, KB, unit,
                                                                     acc, M):
    """BitNet b1.58's path (2-bit TMM codes 00 / 01 / 10, plain, PAIR and QUAD) on the m8 MXU
    built without BIN / CQ / T5: RTL == ISA under AXI stalls."""
    from opentpu import rtlsim
    from opentpu.quant import pack_ternary_mxu
    rng = np.random.default_rng(0xB17 + KB)
    cfg = Config(S=1, D=128, MCOLS=8, ACT_ROWS=16, ACT_BLOCKS=32, DRAM_BYTES=1 << 22)
    D, N = cfg.D, 9
    x = rng.standard_normal((M, KB * D), dtype=np.float32)
    packed = pack_ternary_mxu(rng.choice(np.array([-1, 0, 1], np.int8), (N, KB * D)), D)
    wa, sa, oa, ia, ors = 0x100000, 0x200000, 4096, 0x280000, N + 2
    img = np.zeros(cfg.DRAM_BYTES, np.uint8)
    img[:x.nbytes] = x.view(np.uint8).reshape(-1)
    img[wa:wa + packed.size] = packed.reshape(-1)
    img[sa:sa + 4 * N] = rng.uniform(0.05, 0.2, N).astype(np.float32).view(np.uint8)
    init = rng.standard_normal((M, ors), dtype=np.float32)
    img[ia:ia + init.nbytes] = init.view(np.uint8).reshape(-1)
    s, rs = 0 if unit else sa, packed.shape[1]
    prog = [I.ld(0, 0, M * KB * D)] + ([I.ld(ia, oa, M * ors)] if acc else []) + [
        I.qact(0, M, 0, KB, KB * D, dup4=True),
        I.tmm(wa, s, oa, N, KB, rs, ors, M, 0, 4, unit=unit, rsc=not unit, acc=acc, rmax=True,
              quad=True),
        I.tmm(wa, s, oa + 2 * M * ors, N, KB, rs, ors, M, 0, 4, unit=unit, rsc=not unit),
        I.qact(0, M, 0, KB, KB * D, dup=True),
        I.tmm(wa, s, oa + 4 * M * ors, N, KB, rs, ors, M, 0, 4, unit=unit, rsc=not unit,
              rmax=True, pair=True),
        I.halt()]
    ref = Machine(cfg, [prog], [img.copy()]).run().slices[0]
    drams, tmems, _ = rtlsim.run(cfg, [prog], [img.copy()], axi=True, stall=25, seed=KB,
                                 uarch=OFF)
    np.testing.assert_array_equal(tmems[0], ref.tmem)
    np.testing.assert_array_equal(drams[0], ref.dram)


def test_r8_card_check_area_refusals_on_a_card_without_t5_bin():
    """area_refusals: a board whose CAPS2 lacks TD5 / TBIN / TBGSC refuses all three (the
    host's checks, nothing written); one that has them loads them (the check then fails)."""
    import importlib.util
    from opentpu.host.board import Board
    from opentpu.host.fake import FakeTransport
    spec = importlib.util.spec_from_file_location("r8cc", ROOT / "tools/r8_card_check.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    c4 = Board(FakeTransport(devname=None, regmap=5, ch_bytes=1 << 23, tgsc=True, tgrq=True,
                             tcq=True))
    got = m.area_refusals(c4)
    assert [w for _, w, _ in got] == ["the area build refuses TMM T5",
                                      "the area build refuses TMM BIN",
                                      "the area build refuses TMM BIN GSC"]
    assert all(ok for ok, _, _ in got), got
    c3 = Board(FakeTransport(devname=None, regmap=5, ch_bytes=1 << 23, tgsc=True, tgrq=True,
                             tcq=True, td5=True, tbin=True, tbgsc=True))
    assert not any(ok for ok, _, _ in m.area_refusals(c3))


def test_t5_without_bin_is_not_refused_as_bin():
    """T5's pattern carries the BIN bit: the host's BIN check skips it (T5 has its own, TD5), so
    experiment A (TD5, no TBIN) runs T5; a real BIN TMM is still refused there."""
    import importlib.util
    from opentpu.host.board import Board, ConfigMismatch
    from opentpu.host.fake import FakeTransport
    spec = importlib.util.spec_from_file_location("r8cc", ROOT / "tools/r8_card_check.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    a = Board(FakeTransport(devname=None, regmap=5, ch_bytes=1 << 23, tgsc=True, tgrq=True,
                            tcq=True, td5=True))
    head = [I.ld(0, 0, 128), I.qact(0, 1, 0, 1, 128)]
    a.load(0x3C0000, head + [I.tmm(0x10000, 0x8000, 4000, 4, 5, 128, 4, 1, 0, 20, gsc=True,
                                   dense=True), I.halt()])
    with pytest.raises(ConfigMismatch):
        a.load(0x3C0000, head + [I.tmm(0x10000, 0x8000, 4000, 4, 8, 128, 4, 1, 0, 4, rsc=True,
                                       binary=True), I.halt()])
    got = m.area_refusals(a, m.R8X_LEFT_OUT)
    assert [w for _, w, _ in got] == ["the area build refuses TMM BIN",
                                      "the area build refuses TMM BIN GSC"] and all(o for o, _, _ in got)


def test_r8_card_check_refuses_atan_without_tatan():
    """r5x lean (CFG_HASH add73610): the host refuses TMM T5 and VOP atan on a board whose CAPS2 has
    neither, as BIN / BIN GSC; a board with them loads them (those refusal checks then fail)."""
    import importlib.util
    from opentpu.host.board import Board
    from opentpu.host.fake import FakeTransport
    spec = importlib.util.spec_from_file_location("r8cc", ROOT / "tools/r8_card_check.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.LEFT_OUT[m.R8N_HASH] == ("tbin", "tbgsc", "td5", "tatan")
    n = Board(FakeTransport(devname=None, regmap=5, ch_bytes=1 << 23, tgsc=True, tgrq=True,
                            tcq=True, td5=False))
    got = m.area_refusals(n, m.R8N_LEFT_OUT)
    assert [w for _, w, _ in got] == ["the area build refuses TMM T5",
                                      "the area build refuses TMM BIN",
                                      "the area build refuses TMM BIN GSC",
                                      "the area build refuses VOP atan"] and all(o for o, _, _ in got)
    x = Board(FakeTransport(devname=None, regmap=5, ch_bytes=1 << 23, tgsc=True, tgrq=True,
                            tcq=True, td5=True, tatan=True))
    assert [o for o, _, _ in m.area_refusals(x, m.R8N_LEFT_OUT)] == [False, True, True, False]


def test_no_atan_build_hash():
    """r8_build_lean.cmd's hash: r8 less TBIN TBGSC TD5 TATAN is add73610 (as r8_card_check)."""
    import dataclasses
    from tools.bonsai2_validate import configuration
    from opentpu.target import arch_hash
    caps = {k: k not in ("TBIN", "TBGSC", "TD5", "TATAN") for k in
            "TGSC VCMP TBIN TBGSC VU8 TGRQ VRAND TCQ TATAN QST4 LDX TD5 VBFLY VAFF QST6 VCSWP TGQ4".split()}
    assert arch_hash(dataclasses.replace(configuration(), **caps)) == 0xadd73610
    s = (ROOT / "tools/nextgen/card_runs/r8/r8_build_lean.cmd").read_text()
    assert 'set "OTPU_DROP_FEATURES=TBIN TBGSC TD5 TATAN"' in s and "add73610" in s
    assert "set OTPU_BUILD_TAG=lean" in s and "RAND_W" in s and "g_rom" in s
