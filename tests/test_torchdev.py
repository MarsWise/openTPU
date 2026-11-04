"""PyTorch layers on the device (opentpu/torchdev.py, kernels/linear.py): the linear kernel on
the ISA simulator against float (and the RTL against the ISA simulator), nn.Linear / nn.Conv2d
of a PyTorch module replaced by device kernels with every MAC counted, and a random tiny SmolVLA
(LeRobot) whose actions come out of the device kernels close to its float32 ones."""
import numpy as np
import pytest

from opentpu import Config
from opentpu.kernels import linear as L
from opentpu.runtime import Input, Output, Weight, compile_kernel, run

from conftest import assert_same_state


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _case(D, M, K, N, act="none", seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((M, K)).astype(np.float32)
    w = (rng.standard_normal((N, K)) / np.sqrt(K)).astype(np.float32)
    b = (0.1 * rng.standard_normal(N)).astype(np.float32)
    cfg = Config(S=1, D=D, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16, DRAM_BYTES=1 << 26)
    args = dict(x=Input(L.pad_rows(x, D)), w=Weight(L.pack_weight(w, D)), b=Input(b),
                y=Output((M, N)), M=M, K=K, N=N, act=act)
    return cfg, args, L.reference(x, w, b, act)


@pytest.mark.parametrize("D, M, K, N, act", [(32, 5, 40, 24, "none"), (32, 37, 96, 600, "silu"),
                                             (128, 50, 960, 1300, "none"),
                                             (128, 3, 32, 64, "relu")])
def test_linear_isa_vs_float(D, M, K, N, act):
    cfg, args, want = _case(D, M, K, N, act)
    got = run(*compile_kernel(L.linear, cfg, **args), "isa").outputs["y"]
    assert got.shape == want.shape and rel(got, want) < 0.02, rel(got, want)


def test_linear_wide_k_fits_tmem():
    """K = 12,288 (SmolVLA base's connector): ACT_ROWS rows of it do not fit TMEM's 64 K words,
    so the kernel takes fewer rows a tile (the board's run 2 failed with TMEM exhausted)."""
    rng = np.random.default_rng(5)
    D, M, K, N = 128, 40, 12288, 960
    x = rng.standard_normal((M, K)).astype(np.float32)
    w = (rng.standard_normal((N, K)) / np.sqrt(K)).astype(np.float32)
    b = (0.1 * rng.standard_normal(N)).astype(np.float32)
    cfg = Config(S=1, D=D, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=128, TMEM_WORDS=65536,
                 DRAM_BYTES=1 << 27)
    got = run(*compile_kernel(L.linear, cfg, x=Input(L.pad_rows(x, D)),
                              w=Weight(L.pack_weight(w, D)), b=Input(b), y=Output((M, N)),
                              M=M, K=K, N=N), "isa").outputs["y"]
    want = L.reference(x, w, b)
    assert rel(got, want) < 0.02, rel(got, want)


def test_linear_rtl(have_verilator):
    cfg, args, _ = _case(32, 19, 72, 40, "silu", seed=3)
    comp, imgs = compile_kernel(L.linear, cfg, **args)
    assert_same_state(run(comp, [i.copy() for i in imgs], "isa"),
                      run(comp, [i.copy() for i in imgs], "rtl"))


def _sdpa_case(D, H, KVH, Lq, Lk, dv, seed=0):
    from opentpu.kernels import sdpa as A
    rng = np.random.default_rng(seed)
    dp = A.up(dv, D)

    def mk(n):
        a = np.zeros((n, dp), np.float32)
        a[:, :dv] = rng.standard_normal((n, dv))
        return a
    q, k, v = mk(H * Lq), mk(KVH * Lk), mk(KVH * Lk)
    cfg = Config(S=1, D=D, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16, DRAM_BYTES=1 << 26)
    args = dict(q=Input(q), k=Input(k), v=Input(v), o=Output((Lq, H * dv)),
                kv=Input(np.zeros(A.kv_words(KVH, Lk, dv, D), np.float32)), H=H, KVH=KVH,
                Lq=Lq, Lk=Lk, dv=dv, scale=dv ** -0.5)
    return cfg, args, A.reference(q, k, v, H, KVH, Lq, Lk, dv, dv ** -0.5)


@pytest.mark.parametrize("D, H, KVH, Lq, Lk, dv", [(32, 2, 1, 5, 9, 16), (32, 4, 2, 13, 40, 32),
                                                  (128, 4, 2, 50, 113, 64),
                                                  (128, 15, 5, 8, 300, 64)])
def test_sdpa_isa_vs_float(D, H, KVH, Lq, Lk, dv):
    from opentpu.kernels import sdpa as A
    cfg, args, want = _sdpa_case(D, H, KVH, Lq, Lk, dv)
    got = run(*compile_kernel(A.sdpa, cfg, **args), "isa").outputs["o"]
    assert rel(got, want) < 0.03, rel(got, want)


def test_sdpa_rtl(have_verilator):
    from opentpu.kernels import sdpa as A
    cfg, args, _ = _sdpa_case(32, 4, 2, 6, 37, 32, seed=2)
    comp, imgs = compile_kernel(A.sdpa, cfg, **args)
    assert_same_state(run(comp, [i.copy() for i in imgs], "isa"),
                      run(comp, [i.copy() for i in imgs], "rtl"))


def test_key_prefix_masks_attention():
    """A prefix-LM mask (rows 0-5 see keys 0-5, row 6 all 7, rows 7-9 all 10) runs as three
    device calls and equals torch's masked softmax attention."""
    torch = pytest.importorskip("torch")
    from opentpu import torchdev as TD
    Lq = Lk = 10
    n = np.array([6] * 6 + [7] + [10] * 3)
    mask = np.arange(Lk)[None, :] < n[:, None]
    assert [k for k, _ in TD.key_prefixes(mask)] == [6, 7, 10]
    assert TD.key_prefixes(~mask) is None
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, Lq, 4, 32, generator=g), torch.randn(1, Lk, 2, 32, generator=g),
               torch.randn(1, Lk, 2, 32, generator=g))
    r, cov = TD.Runner("isa", cfg=Config(S=1, D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16,
                                         DRAM_BYTES=1 << 24)), TD.Coverage()
    att = TD.DeviceAttention(r, cov, 4, 2)
    got = att(torch.from_numpy(mask)[None], 1, 32, q, k, v)[0].numpy()
    kk, vv = k.repeat_interleave(2, 2), v.repeat_interleave(2, 2)
    s = torch.einsum("lhd,mhd->hlm", q[0], kk[0]) * 32 ** -0.5
    s = s.masked_fill(~torch.from_numpy(mask)[None], float("-inf")).softmax(-1)
    want = torch.einsum("hlm,mhd->lhd", s, vv[0]).reshape(Lq, -1).numpy()
    assert rel(got, want) < 0.03 and r.stats["runs"] == 3


def test_torch_module_on_the_device():
    torch = pytest.importorskip("torch")
    from opentpu import torchdev as TD
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 4, stride=4), torch.nn.Flatten(1),
                            torch.nn.Linear(16 * 4 * 4, 64), torch.nn.GELU(),
                            torch.nn.Linear(64, 10)).eval()
    x = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        want = m(x)
    r, cov = TD.Runner("isa", cfg=Config(S=1, D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16,
                                         DRAM_BYTES=1 << 24)), TD.Coverage()
    assert TD.to_device(m, r, cov) == 3
    with torch.no_grad(), TD.host_counter(cov):
        got = m(x)
    assert rel(got.numpy(), want.numpy()) < 0.03
    assert cov.total("host") == 0
    assert cov.total("device") == 2 * (16 * 48 * 16) + 2 * 256 * 64 + 2 * 64 * 10
    assert r.stats["runs"] == 4                       # two images' convs, two linears


def test_tiny_smolvla_actions():
    pytest.importorskip("torch")
    pytest.importorskip("lerobot.policies.smolvla.modeling_smolvla",
                        reason="LeRobot (SmolVLA) not installed")
    import transformers
    if tuple(int(v) for v in transformers.__version__.split(".")[:2]) >= (5, 6):
        pytest.skip("LeRobot's SmolVLA needs transformers < 5.6 (its pin)")
    import copy
    import sys
    from pathlib import Path

    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import smolvla_run as SV

    from opentpu import torchdev as TD
    pol, cfg = SV.tiny_policy(0)
    x = SV.inputs(cfg, 0, 1000)
    with torch.no_grad():
        ref = pol.model.sample_actions(*x[:5], noise=x[5].clone())
    dev = copy.deepcopy(pol)
    r, cov = TD.Runner("isa"), TD.Coverage()
    TD.to_device(dev, r, cov)
    vwe = dev.model.vlm_with_expert
    att = TD.DeviceAttention(r, cov, vwe.num_attention_heads, vwe.num_key_value_heads)
    vwe.eager_attention_forward = att
    TD.hf_attention(vwe.vlm.model.vision_model, att)
    with torch.no_grad(), TD.host_counter(cov):
        got = dev.model.sample_actions(*x[:5], noise=x[5].clone())
    a, b = got.numpy()[0, :, :6], ref.numpy()[0, :, :6]
    assert rel(a, b) < 0.05
    assert cov.total("host") == 0 and cov.total("device") > 0     # every MAC on the device


def test_a_linear_too_long_for_imem_runs_in_row_chunks():
    """The linear kernel's program grows with M: a layer whose program would not fit IMEM runs
    its rows in chunks that do (r4: SmolVLA base's were 219140 instructions into 4096, and ran
    overwritten until the loader refused them), the result the one-program run's bit for bit;
    a kernel that cannot be split raises ProgramTooLong instead of running."""
    torch = pytest.importorskip("torch")
    from opentpu import torchdev as TD
    torch.manual_seed(1)
    lin = torch.nn.Linear(96, 200)
    x = torch.randn(300, 96)
    outs, runs = [], []
    for imem in (1 << 20, 8 * 120):
        r = TD.Runner("isa", cfg=Config(S=1, D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16,
                                        DRAM_BYTES=1 << 24, IMEM_WORDS=imem))
        dl = TD.DeviceLinear(lin, r, TD.Coverage())
        outs.append(dl(x).detach().numpy())
        runs.append(r.stats["runs"])
    assert runs[0] == 1 and runs[1] > 1, runs
    assert np.array_equal(outs[0].view(np.uint32), outs[1].view(np.uint32))
    r = TD.Runner("isa", cfg=Config(S=1, D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16,
                                    DRAM_BYTES=1 << 24, IMEM_WORDS=8 * 4))
    with pytest.raises(TD.ProgramTooLong, match="IMEM holds 4"):
        TD.DeviceLinear(lin, r, TD.Coverage())(x)


def test_resident_weights_are_written_once_and_change_nothing():
    """Runner(resident=True), the default: a layer's weights are quantized and placed once, above
    every call's data, and each later call moves only activations and the program; the outputs
    are the non-resident run's bit for bit (the same quantized bytes, other addresses)."""
    torch = pytest.importorskip("torch")
    from opentpu import torchdev as TD
    torch.manual_seed(2)
    m = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 4, stride=4), torch.nn.Flatten(1),
                            torch.nn.Linear(16 * 4 * 4, 64), torch.nn.GELU(),
                            torch.nn.Linear(64, 10)).eval()
    xs = [torch.randn(2, 3, 16, 16) for _ in range(3)]
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16, DRAM_BYTES=1 << 27)
    outs, runners = {}, {}
    for res in (False, True):
        mm = __import__("copy").deepcopy(m)
        r = TD.Runner("isa", cfg=cfg, resident=res)
        TD.to_device(mm, r, TD.Coverage())
        with torch.no_grad():
            outs[res] = [mm(x).numpy() for x in xs]
        runners[res] = r
    for a, b in zip(outs[False], outs[True]):
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    r0, r1 = runners[False], runners[True]
    assert r0.weight_bytes == 0 and r1.weight_bytes > 0
    assert len(r1._res) == 3                         # the conv and the two linears, once each
    assert r1.call_bytes < r0.call_bytes - 2 * r1.weight_bytes   # 3 passes: weights not re-sent
    assert cfg.DRAM_BYTES // 2 <= r1._heap < cfg.DRAM_BYTES


def test_programs_are_traced_once_per_distinct_call(monkeypatch):
    """Runner caches each distinct call's programs (the same layer, shapes and resident weights):
    a repeated forward traces nothing new, its outputs bit for bit the uncached run's; with
    OTPU_PROG_CACHE_CHECK=1 every reuse is checked against a fresh trace."""
    torch = pytest.importorskip("torch")
    from opentpu import torchdev as TD
    torch.manual_seed(3)
    m = torch.nn.Sequential(torch.nn.Linear(40, 64), torch.nn.GELU(), torch.nn.Linear(64, 10)).eval()
    xs = [torch.randn(5, 40) for _ in range(3)]
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16, DRAM_BYTES=1 << 24)
    outs = {}
    for cache in (False, True):
        mm = __import__("copy").deepcopy(m)
        r = TD.Runner("isa", cfg=cfg, cache_programs=cache)
        TD.to_device(mm, r, TD.Coverage())
        with torch.no_grad():
            outs[cache] = [mm(x).numpy() for x in xs]
        if cache:
            assert r.stats["traced"] == 2 and r.stats["cached"] == 4, r.stats
        else:
            assert r.stats["traced"] == 0
    for a, b in zip(outs[False], outs[True]):
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    monkeypatch.setenv("OTPU_PROG_CACHE_CHECK", "1")
    r = TD.Runner("isa", cfg=cfg)
    TD.to_device(m, r, TD.Coverage())
    with torch.no_grad():
        for x in xs:
            m(x)


def test_attention_too_long_for_imem_runs_in_row_and_head_chunks():
    """An attention whose program does not fit IMEM (SmolVLA base on r5-m8: sdpa 111,620
    instructions into 4096) runs in query-row chunks, and at one row in KV-head groups; every
    piece is the same attention, so the result equals the one-program run's bit for bit."""
    torch = pytest.importorskip("torch")
    from opentpu import torchdev as TD
    g = torch.Generator().manual_seed(3)
    Lq, Lk = 12, 40
    q, k, v = (torch.randn(1, Lq, 4, 32, generator=g), torch.randn(1, Lk, 2, 32, generator=g),
               torch.randn(1, Lk, 2, 32, generator=g))
    mask = torch.ones(1, Lq, Lk, dtype=torch.bool)
    outs, runs = [], []
    # 289 instructions in one program; 200: row chunks; 80: one row is 110, a KV head's 56
    for imem in (1 << 20, 8 * 200, 8 * 80):
        r = TD.Runner("isa", cfg=Config(S=1, D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16,
                                        DRAM_BYTES=1 << 24, IMEM_WORDS=imem))
        try:
            outs.append(TD.DeviceAttention(r, TD.Coverage(), 4, 2)(mask, 1, 32, q, k, v).numpy())
        except TD.ProgramTooLong:
            outs.append(None)
        runs.append(r.stats["runs"])
    assert runs[0] == 1 and outs[0] is not None
    for o, n in zip(outs[1:], runs[1:]):
        assert o is not None and n > 1, runs
        assert np.array_equal(o.view(np.uint32), outs[0].view(np.uint32)), n
    assert runs[2] > runs[1]
