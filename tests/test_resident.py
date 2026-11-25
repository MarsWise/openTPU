"""Resident graph ops (opentpu/resident.py) against PyTorch, on the ISA simulator."""
import numpy as np
import pytest

from opentpu import resident as RG
from opentpu.isasim import Config

torch = pytest.importorskip("torch")


def cfg():
    return Config(D=128, MCOLS=4, ACT_ROWS=8, TFN=True, DSTEP=True, STREAM=True,
                  DRAM_BYTES=1 << 26)


def run(g, inputs, outs):
    progs = g.compile()
    img = g.image()
    for t, x in inputs:
        g.put(img, t, x)
    img, _ = RG.run_isa(g.cfg, progs, img)
    return [g.read(img, t) for t in outs]


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("cin,cout,k,s,p", [(2, 64, 5, 1, 2), (64, 64, 3, 2, 1),
                                            (96, 32, 5, 1, 2), (32, 16, 1, 1, 0)])
def test_conv1d(cin, cout, k, s, p):
    torch.manual_seed(cin + k)
    m = torch.nn.Conv1d(cin, cout, k, s, p)
    x = torch.randn(1, cin, 16)
    want = m(x)[0].T.detach().numpy()
    g = RG.Graph(cfg())
    xt = g.tensor(16, cin, 2)
    y = g.conv1d(xt, m.weight.detach().numpy(), m.bias.detach().numpy(), s, p)
    got, = run(g, [(xt, x[0].T.numpy())], [y])
    assert got.shape == want.shape and rel(got, want) < 0.02


def test_conv_transpose_groupnorm_mish_film_cat_linear():
    torch.manual_seed(0)
    ct = torch.nn.ConvTranspose1d(64, 64, 4, 2, 1)
    gn = torch.nn.GroupNorm(8, 64)
    with torch.no_grad():
        gn.weight.normal_(1, .2)
        gn.bias.normal_(0, .2)
    lin = torch.nn.Linear(40, 128)
    x = torch.randn(1, 64, 8)
    c = torch.randn(1, 40)
    skip = torch.randn(1, 64, 16)
    with torch.no_grad():
        u = ct(x)
        h = torch.nn.functional.mish(gn(u))
        e = lin(torch.nn.functional.mish(c))
        f = h * e[:, :64, None] + e[:, 64:, None]
        want = torch.cat([f, skip], 1)[0].T.numpy()
    g = RG.Graph(cfg())
    xt, ct_in, sk = g.tensor(8, 64, 2), g.tensor(1, 40), g.tensor(16, 64, 2)
    u_t = g.conv_transpose1d(xt, ct.weight.detach().numpy(), ct.bias.detach().numpy(), 2, 1)
    h_t = g.group_norm(u_t, 8, gn.weight.detach().numpy(), gn.bias.detach().numpy(), 1e-5,
                       act="mish")
    cm = g.unary(ct_in, "mish")
    e_t = g.linear(cm, lin.weight.detach().numpy(), lin.bias.detach().numpy())
    g.film(h_t, e_t.rows(0, 1, 0, 64).__class__(e_t.addr, (64,), (1,)),
           e_t.rows(0, 1, 0, 64).__class__(e_t.addr + 4 * 64, (64,), (1,)))
    cat = g.tensor(16, 128, 2)
    g.copy(h_t, cat.cols(0, 64))
    g.copy(sk, cat.cols(64, 128))
    got, = run(g, [(xt, x[0].T.numpy()), (ct_in, c.numpy()), (sk, skip[0].T.numpy())], [cat])
    assert rel(got, want) < 0.03, rel(got, want)


def test_diffusion_policy_denoising_loop_resident():
    """The tiny policy's whole DDPM loop (U-Net, timestep MLP, FiLM, updates) on the device,
    against the policy's own float loop with the same condition and noise."""
    pytest.importorskip("lerobot")
    pytest.importorskip("diffusers")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from diffusion_run import tiny_policy
    from opentpu import diffusion_resident as DR
    pol, pc = tiny_policy(0)
    dm = pol.diffusion
    rng = np.random.default_rng(0)
    gcd = dm.unet.down_modules[0][0].cond_encoder[1].in_features - \
        dm.unet.diffusion_step_encoder[0].dim
    gc = rng.normal(size=gcd).astype(np.float32)
    n0 = rng.normal(size=(pc.horizon, 2)).astype(np.float32)
    ref, zs = DR.reference_noise(pol, gc, n0, 3, seed=1)
    loop = DR.build(pol, cfg(), 3)
    img, _ = RG.run_isa(loop.g.cfg, loop.programs(), loop.image(gc, n0, zs))
    assert rel(loop.actions(img), ref) < 0.02


def test_diffusion_policy_noise_on_the_device():
    """FPGA-only: the host writes a seed, the device draws the initial noise and each step's
    (device_noise: VOP RAND, Box-Muller); the actions against the policy's own float loop
    given those noises (Loop.noises: the device's values, bit for bit)."""
    pytest.importorskip("lerobot")
    pytest.importorskip("diffusers")
    import dataclasses
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from diffusion_run import tiny_policy
    from opentpu import diffusion_resident as DR
    pol, pc = tiny_policy(0)
    dm = pol.diffusion
    rng = np.random.default_rng(0)
    gcd = dm.unet.down_modules[0][0].cond_encoder[1].in_features - \
        dm.unet.diffusion_step_encoder[0].dim
    gc = rng.normal(size=gcd).astype(np.float32)
    loop = DR.build(pol, dataclasses.replace(cfg(), VRAND=True), 3, device_noise=True)
    img, _ = RG.run_isa(loop.g.cfg, loop.programs(), loop.image(gc, None, None, seed=7))
    z0, zs = loop.noises(7)
    ref = DR.reference_given(pol, gc, z0, zs, 3)
    assert rel(loop.actions(img), ref) < 0.02


def _graph_2d():
    torch.manual_seed(5)
    conv = torch.nn.Conv2d(3, 16, 3, 2, 1)
    gn = torch.nn.GroupNorm(4, 16)
    kp = torch.nn.Conv2d(16, 4, 1)
    with torch.no_grad():
        gn.weight.normal_(1, .2)
        gn.bias.normal_(0, .2)
    x = torch.randn(1, 3, 12, 10)
    c = cfg()
    g = RG.Graph(c)
    xi = g.tensor2(12, 10, 3, 1, c.D)
    y = g.conv2d(xi, conv.weight.detach().numpy(), conv.bias.detach().numpy(), 2, 1)
    y = g.group_norm2d(y, 4, gn.weight.detach().numpy(), gn.bias.detach().numpy(), 1e-5, "relu")
    p = g.maxpool2d(y, 3, 2, 1)
    s = g.conv2d(p, kp.weight.detach().numpy(), kp.bias.detach().numpy(), 1, 0)
    H, W = s.H, s.W
    px, py = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    out = g.tensor(1, 8)
    g.op("ss", RG.spatial_softmax_k, x=s.flat(), px=tuple(px.reshape(-1)),
         py=tuple(py.reshape(-1)), y=out.rows())
    with torch.no_grad():
        f = kp(torch.nn.functional.max_pool2d(torch.relu(gn(conv(x))), 3, 2, 1))
        a = torch.softmax(f.reshape(4, -1), -1).numpy()
    want = np.concatenate([a @ px.reshape(-1), a @ py.reshape(-1)])
    img = g.image()
    fl = img.view(np.float32)
    base = xi.interior().base
    for h in range(12):
        o = (int(base.const) if hasattr(base, "const") else int(base)) // 4 + h * xi.rs
        fl[o:o + 30] = x[0, :, h].T.reshape(-1).numpy()
    return g, img, out, want


def test_resident_2d_encoder_ops():
    g, img, out, want = _graph_2d()
    img, _ = RG.run_isa(g.cfg, g.compile(), img)
    got = g.read(img, out)[0]
    assert np.abs(got - want).max() < 0.03, (got, want)


def test_resident_ops_rtl(have_verilator):
    """Every program of the 2-D graph (conv, looped GroupNorm, max-pool, soft-argmax) and of a
    1-D one (strided-view conv, zero insertion, transposed conv, GroupNorm + Mish) on the RTL
    == the ISA, the image carried from program to program."""
    import dataclasses
    from opentpu import rtlsim
    from opentpu.isasim import Machine
    g, img, _, _ = _graph_2d()
    g1 = RG.Graph(cfg())
    xt = g1.tensor(8, 64, 2)
    u = g1.conv_transpose1d(xt, np.random.default_rng(1).normal(0, .1, (64, 32, 4))
                            .astype(np.float32), None, 2, 1)
    g1.group_norm(u, 8, np.ones(32), np.zeros(32), 1e-5, act="mish")
    img1 = g1.image()
    g1.put(img1, xt, np.random.default_rng(2).normal(size=(8, 64)))
    for gr, im in ((g, img), (g1, img1)):
        c = dataclasses.replace(gr.cfg, DRAM_BYTES=1 << max(20, (len(im) - 1).bit_length()))
        im = np.pad(im, (0, c.DRAM_BYTES - len(im)))
        for i, (name, p) in enumerate(gr.compile()):
            ref = Machine(c, [p], [im.copy()]).run().slices[0].dram
            drams, _, _ = rtlsim.run(c, [p], [im.copy()], axi=True, stall=30, seed=i)
            np.testing.assert_array_equal(drams[0], ref, err_msg=name)
            im = ref


@pytest.mark.parametrize("H,KVH,dh", [(4, 2, 64), (3, 1, 32)])
def test_masked_attention_groups(H, KVH, dh):
    rng = np.random.default_rng(H)
    Lq, Lk, P = 12, 20, 8
    q = rng.normal(size=(Lq, H * dh)).astype(np.float32)
    k = rng.normal(size=(Lk, KVH * dh)).astype(np.float32)
    v = rng.normal(size=(Lk, KVH * dh)).astype(np.float32)
    # rows 0..3 see the 7-key prefix-LM block, row 4 all 8 prefix keys, then causal suffix rows
    groups = [(0, 4, 7), (4, 5, 8)] + [(r, r + 1, P + r - 4) for r in range(5, Lq)]
    g = RG.Graph(cfg())
    qt, kt, vt = g.tensor(Lq, H * dh), g.tensor(Lk, KVH * dh), g.tensor(Lk, KVH * dh)
    o = g.attention(qt, kt, vt, H, KVH, dh, groups)
    got, = run(g, [(qt, q), (kt, k), (vt, v)], [o])
    want = np.zeros_like(got)
    for r0, r1, n in groups:
        for h in range(H):
            gq = h // (H // KVH)
            s = q[r0:r1, h * dh:(h + 1) * dh] @ k[:n, gq * dh:(gq + 1) * dh].T * dh ** -0.5
            p = np.exp(s - s.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            want[r0:r1, h * dh:(h + 1) * dh] = p @ v[:n, gq * dh:(gq + 1) * dh]
    assert rel(got, want) < 0.03, rel(got, want)


def test_smolvla_resident_tiny():
    """SmolVLA (vision, connector, prefix layers, 10 Euler steps of the expert) on the device
    against the policy's own float sample_actions."""
    pytest.importorskip("lerobot")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from smolvla_run import inputs, tiny_policy
    from opentpu import smolvla_resident as SR
    pol, pc = tiny_policy(0)
    imgs, masks, toks, lmask, state, noise = inputs(pc, 0, 1000)
    with torch.no_grad():
        ref = pol.model.sample_actions(imgs, masks, toks, lmask, state,
                                       noise=noise.clone())[0].numpy()
    vla = SR.build(pol, cfg(), toks[0].numpy())
    img = vla.image([i[0].numpy() for i in imgs], state[0].numpy(), noise[0].numpy())
    img, _ = RG.run_isa(vla.g.cfg, vla.programs(), img)
    assert rel(vla.actions(img), ref) < 0.03


def test_smolvla_noise_on_the_device():
    """FPGA-only SmolVLA: the host writes a seed, the device draws the flow's initial noise
    (device_noise); the actions against sample_actions given that noise (VLA.noise)."""
    pytest.importorskip("lerobot")
    import dataclasses
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from smolvla_run import inputs, tiny_policy
    from opentpu import smolvla_resident as SR
    pol, pc = tiny_policy(0)
    imgs, masks, toks, lmask, state, _ = inputs(pc, 0, 1000)
    vla = SR.build(pol, dataclasses.replace(cfg(), VRAND=True), toks[0].numpy(),
                   device_noise=True)
    noise = torch.as_tensor(vla.noise(11))[None]
    with torch.no_grad():
        ref = pol.model.sample_actions(imgs, masks, toks, lmask, state,
                                       noise=noise.clone())[0].numpy()
    img = vla.image([i[0].numpy() for i in imgs], state[0].numpy(), None, seed=11)
    img, _ = RG.run_isa(vla.g.cfg, vla.programs(), img)
    assert rel(vla.actions(img), ref) < 0.03


def test_resident_transformer_ops_rtl(have_verilator):
    """Masked attention (prefix-LM + causal groups), RoPE, RMSNorm, row moves: RTL == ISA."""
    import dataclasses
    from opentpu import rtlsim
    from opentpu.isasim import Machine
    rng = np.random.default_rng(9)
    g = RG.Graph(cfg())
    x = g.tensor(10, 128)
    n = g.rms_norm(x, np.ones(128), 1e-5)
    q = g.linear(n, rng.normal(0, .1, (128, 128)), None)
    k = g.linear(n, rng.normal(0, .1, (64, 128)), None)
    g.rope(q, 4, 32, np.arange(10))
    g.rope(k, 2, 32, np.arange(10))
    o = g.attention(q, k, k, 4, 2, 32, [(0, 6, 6), (6, 7, 7)] + [(r, r + 1, r + 1)
                                                               for r in range(7, 10)])
    y = g.tensor(4, 256)
    g.copy_rows(o.rows(), y.rows(), [(9, 0, 128), (0, 3, 0), (5, 1, 64)], 64)
    img = g.image()
    g.put(img, x, rng.normal(size=(10, 128)))
    c = dataclasses.replace(g.cfg, DRAM_BYTES=1 << max(20, (len(img) - 1).bit_length()))
    img = np.pad(img, (0, c.DRAM_BYTES - len(img)))
    for i, (name, p) in enumerate(g.compile()):
        ref = Machine(c, [p], [img.copy()]).run().slices[0].dram
        drams, _, _ = rtlsim.run(c, [p], [img.copy()], axi=True, stall=30, seed=i)
        np.testing.assert_array_equal(drams[0], ref, err_msg=name)
        img = ref


def test_elementwise_ops_split_rows_past_tmem():
    """add / unary / copy / film over a tensor larger than TMEM: one program per row block
    (SmolVLA base's vision residual [1024, 768] on r8's 131072-word TMEM), exact vs numpy."""
    import dataclasses
    c = dataclasses.replace(cfg(), TMEM_WORDS=4096)      # cap 1024 words: 300 x 16 = 4800
    rng = np.random.default_rng(3)
    L, C = 300, 16
    a, b = rng.standard_normal((2, L, C)).astype(np.float32)
    sc, bi = rng.standard_normal((2, C)).astype(np.float32)
    g = RG.Graph(c)
    at, bt = g.tensor(L, C, 1), g.tensor(L, C, 1)
    s = g.add(at, bt, name="res")
    u = g.unary(s, "relu")
    cp = g.copy(u, g.tensor(L, C, 1))
    g.film(cp, g.vec(sc), g.vec(bi))
    names = [n for n, _, _ in g.ops]
    assert sum(n.startswith("res.") for n in names) == -(-L // (1024 // C))
    got, s_got = run(g, [(at, a), (bt, b)], [cp, s])
    assert np.array_equal(s_got, a + b)
    assert rel(got, np.maximum(a + b, 0) * sc + bi) < 1e-6


def test_long_attention_split_is_identical(monkeypatch):
    """A long attention (Graph.attention past ATTN_KV_RUNS / ATTN_SPLIT_ROWS: KV quantization
    programs, then query programs of a few row blocks) gives the single program's output bit
    for bit; groups with prefix masks and causal rows."""
    rng = np.random.default_rng(7)
    H, KVH, dh, Lq, Lk = 4, 2, 64, 40, 48
    q = rng.normal(size=(Lq, H * dh)).astype(np.float32)
    k = rng.normal(size=(Lk, KVH * dh)).astype(np.float32)
    v = rng.normal(size=(Lk, KVH * dh)).astype(np.float32)
    groups = [(0, 30, 33), (30, 31, 40)] + [(r, r + 1, r + 9) for r in range(31, Lq)]
    out, nprog = [], []
    for split in (False, True):
        if split:
            monkeypatch.setattr(RG, "ATTN_KV_RUNS", 4)
            monkeypatch.setattr(RG, "ATTN_SPLIT_ROWS", 8)
            monkeypatch.setattr(RG, "ATTN_QBLOCK_RUNS", 1)
        g = RG.Graph(cfg())
        qt, kt, vt = g.tensor(Lq, H * dh), g.tensor(Lk, KVH * dh), g.tensor(Lk, KVH * dh)
        o = g.attention(qt, kt, vt, H, KVH, dh, groups)
        nprog.append(len(g.ops))
        got, = run(g, [(qt, q), (kt, k), (vt, v)], [o])
        out.append(got)
    assert nprog[1] > nprog[0]
    assert np.array_equal(out[0].view(np.uint32), out[1].view(np.uint32))


def test_siglip_sized_attention_fits_imem():
    """SmolVLA base's vision attention (1024 patches, 12 heads of 64, one bidirectional group)
    on r8's configuration: every program of the split attention within IMEM (one program was
    154758 instructions)."""
    from opentpu.isasim import board_config
    c = board_config(MCOLS=8, TMEM_WORDS=131072, DRAM_BYTES=1 << 26)
    g = RG.Graph(c)
    qt, kt, vt = g.tensor(1024, 768), g.tensor(1024, 768), g.tensor(1024, 768)
    g.attention(qt, kt, vt, 12, 12, 64, [(0, 1024, 1024)])
    progs = g.compile()
    assert len(progs) > 1 and max(len(p) for _, p in progs) <= c.IMEM_WORDS // 8


def test_linear_over_many_rows_is_split_and_identical(monkeypatch):
    """Graph.linear past LINEAR_ROWS: a program per row block (SmolVLA base's vision fc1 over
    1024 rows was 8707 instructions, IMEM 4096), the output that of one program bit for bit."""
    rng = np.random.default_rng(11)
    x = rng.normal(size=(40, 96)).astype(np.float32)
    w = rng.normal(size=(64, 96)).astype(np.float32)
    b = rng.normal(size=64).astype(np.float32)
    out = []
    for rows in (256, 16):
        monkeypatch.setattr(RG, "LINEAR_ROWS", rows)
        g = RG.Graph(cfg())
        xt = g.tensor(40, 96)
        y = g.linear(xt, w, b, act="relu")
        assert len(g.ops) == -(-40 // rows)
        out.append(run(g, [(xt, x)], [y])[0])
    assert np.array_equal(out[0].view(np.uint32), out[1].view(np.uint32))


@pytest.mark.parametrize("workers", [1, 2])
def test_run_board_entries_once_per_distinct_program(workers):
    """run_board computes the wait entries before the run, once per distinct program (a loop's
    repeated steps share theirs), in a process pool or not: each load gets exactly waits.entries'
    entries of its program, the programs in order."""
    from opentpu import isa as I
    from opentpu import waits as W
    torch.manual_seed(3)
    m = torch.nn.Conv1d(64, 64, 3, 1, 1)
    g = RG.Graph(cfg())
    xt = g.tensor(16, 64, 2)
    y = g.conv1d(xt, m.weight.detach().numpy(), m.bias.detach().numpy(), 1, 1)
    g.conv1d(y, m.weight.detach().numpy(), m.bias.detach().numpy(), 1, 1)
    progs = g.compile()
    progs = progs + progs + progs + progs + progs        # five "steps" of the same programs (> 8)

    class FakeBoard:
        waitc, slices, _info = True, 1, {"D": 128}

        def __init__(self):
            self.loads = []

        def load_program(self, addr, words, entries):
            self.loads.append((addr, np.array(words), entries))

        def run(self, timeout=None):
            return {"cycles": 10, "instructions": [len(self.loads[-1][1]) // 8]}

    W._mem.clear()
    b = FakeBoard()
    st = RG.run_board(b, progs, 4096, workers=workers)
    assert st["runs"] == len(progs) and st["distinct"] == len(progs) // 5
    for (_, p), (addr, words, e) in zip(progs, b.loads):
        w = np.asarray(I.assemble(p), np.uint32)
        assert addr == 4096 and np.array_equal(words, w)
        assert np.array_equal(e, W.entries(w, 128, 1, prog=p))
    b2 = FakeBoard()
    b2.waitc = False
    RG.run_board(b2, progs, 4096, workers=workers)
    assert all(e is None for _, _, e in b2.loads)


def _chain_graph(reps: int = 3):
    """Conv1d, GroupNorm + Mish, conv1d: three programs, `reps` times (a loop's steps)."""
    torch.manual_seed(3)
    m = torch.nn.Conv1d(64, 64, 3, 1, 1)
    w, b = m.weight.detach().numpy(), m.bias.detach().numpy()
    g = RG.Graph(cfg())
    xt = g.tensor(16, 64, 2)
    y = g.conv1d(xt, w, b, 1, 1)
    y = g.group_norm(y, 8, np.ones(64), np.zeros(64), 1e-5, act="mish")
    g.conv1d(y, w, b, 1, 1)
    img = g.image()
    g.put(img, xt, np.random.default_rng(1).normal(size=(16, 64)))
    return g, g.compile() * reps, img


def test_chained_run_isa_identical():
    """The programs as one chained run (chain_slots: HALT CHAIN from slot to slot) leave the
    image the sequential runs leave, bit for bit; two instructions more a link."""
    g, progs, img = _chain_graph()
    a, na = RG.run_isa(g.cfg, progs, img.copy())
    at = -(-(len(img) + (1 << 16)) // 4096) * 4096
    b, nb = RG.run_isa_chained(g.cfg, progs, img.copy(), at)
    assert np.array_equal(a, b)
    assert nb == na + 2 * (len(progs) - 1)


class _ChainCard:
    """A card model for run_board_chained: write / load_program / run, the run on the ISA
    simulator (it follows HALT CHAIN through the slots' words)."""
    slices = 1

    def __init__(self, c, waitc):
        self.c, self.waitc = c, waitc
        self._info = {"D": c.D, "cfg": {"IMEM_WORDS": c.IMEM_WORDS,
                                        "DRAM_BYTES": c.DRAM_BYTES}}
        self.dram = np.zeros(c.DRAM_BYTES, np.uint8)
        self.loads, self.runs = [], 0

    def write(self, addr, data):
        data = np.asarray(data).view(np.uint8)
        self.dram[addr:addr + len(data)] = data

    def load_program(self, addr, words, entries=None):
        self.loads.append((addr, np.asarray(words, np.uint32), entries))

    def run(self, timeout=None):
        from opentpu import isa as I
        from opentpu.isasim import Machine
        self.runs += 1
        _, w, _ = self.loads[-1]
        prog = [I.Instr.decode(w[8 * k:8 * k + 8]) for k in range(len(w) // 8)]
        m = Machine(self.c, [prog], [self.dram]).run(max_steps=1 << 40)
        self.dram = m.slices[0].dram
        return {"cycles": 1, "instructions": [m.slices[0].icount]}


@pytest.mark.parametrize("waitc", [False, True])
def test_run_board_chained_one_launch(waitc):
    """run_board_chained: the slots written at once (each its program with the chain tail and,
    on a WAITC card, its wait entries: the program's own, computed once per distinct program,
    and zeros for the tail), the first program loaded and run once; the card (an ISA model
    that follows HALT CHAIN through the slots' words) leaves the sequential runs' image."""
    import dataclasses
    from opentpu import isa as I
    from opentpu import waits as W
    from opentpu.isasim import Machine
    g, progs, img = _chain_graph()
    ref, _ = RG.run_isa(g.cfg, progs, img.copy())
    c = dataclasses.replace(g.cfg, DRAM_BYTES=1 << 22)

    W._mem.clear()
    card = _ChainCard(c, waitc)
    card.write(0, img)
    at = -(-(len(img) + (1 << 16)) // 4096) * 4096
    st = RG.run_board_chained(card, progs, at, workers=1)
    assert card.runs == 1 and len(card.loads) == 1 and card.loads[0][0] == at
    assert st["programs"] == len(progs) and st["distinct"] == len(progs) // 3
    assert np.array_equal(card.dram[:len(img)], ref)
    # the slots: each program's words with its tail, then (WAITC) its own entries + 3 zeros
    words = [np.asarray(I.assemble(p), np.uint32) for _, p in progs]
    ents = [W.entries(w, c.D, 1, prog=p) for w, (_, p) in zip(words, progs)] if waitc \
        else [None] * len(words)
    blob, slots, loaded = RG.chain_slots(words, ents, at, c.D, c.IMEM_WORDS)
    assert np.array_equal(card.dram[at:at + 4 * len(blob)].view(np.uint32), blob)
    for i, ((addr, n), (w, e)) in enumerate(zip(slots, loaded)):
        assert addr % c.D == 0 and len(w) == 8 * n
        assert np.array_equal(w[:len(words[i]) - 8], words[i][:-8])
        if waitc:
            k = len(words[i]) // 8 - 1
            assert np.array_equal(e[:k], ents[i][:k]) and not e[k:].any()
            o = (addr - at) // 4
            assert np.array_equal(blob[o:o + len(W.loadable(w, e, c.D))], W.loadable(w, e, c.D))
        else:
            assert e is None


@pytest.mark.parametrize("seq_wc", [0, 1])
def test_chained_run_rtl(have_verilator, seq_wc):
    """The chained run on the board RTL (the loader's HALT CHAIN from DRAM slots; with the
    WAITC sequencer each slot's wait entries after its words) == the sequential ISA runs."""
    import dataclasses
    from opentpu import isa as I
    from opentpu import rtlsim
    from opentpu import waits as W
    g, progs, img = _chain_graph(2)
    ref, _ = RG.run_isa(g.cfg, progs, img.copy())
    c = dataclasses.replace(g.cfg, DRAM_BYTES=1 << 22)
    at = -(-(len(img) + (1 << 16)) // 4096) * 4096
    words = [np.asarray(I.assemble(p), np.uint32) for _, p in progs]
    ents = [W.entries(w, c.D, 1, prog=p) for w, (_, p) in zip(words, progs)] if seq_wc \
        else [None] * len(words)
    blob, slots, loaded = RG.chain_slots(words, ents, at, c.D, c.IMEM_WORDS)
    im = np.zeros(at + 4 * len(blob), np.uint8)
    im[:len(img)] = img
    im[at:] = blob.view(np.uint8)
    first = [I.Instr.decode(loaded[0][0][8 * k:8 * k + 8]) for k in range(slots[0][1])]
    ua = {"SEQ_WC": seq_wc, "WIN": 16} if seq_wc else None
    drams, _, _ = rtlsim.run(c, [first], [im], axi=True, stall=30, seed=1, uarch=ua,
                             boot=True)
    np.testing.assert_array_equal(drams[0][:len(img)], ref)


@pytest.mark.parametrize("waitc", [False, True])
def test_run_board_chained_segments(waitc):
    """Slots that do not fit between `at` and the DRAM's end: consecutive segments that do
    (chain_segments), one launch each over the same slot area; the image still the sequential
    runs'. A program larger than the room: refused before anything is written."""
    import dataclasses
    from opentpu import isa as I
    from opentpu import waits as W
    g, progs, img = _chain_graph()
    ref, _ = RG.run_isa(g.cfg, progs, img.copy())
    at = -(-(len(img) + (1 << 16)) // 4096) * 4096
    words = [np.asarray(I.assemble(p), np.uint32) for _, p in progs]
    big = max(len(w) for w in words) // 8 + 2
    one = -(-32 * big // 128) * 128 + (8 * big if waitc else 0)
    c = dataclasses.replace(g.cfg, DRAM_BYTES=at + 2 * one + 4096)     # about two slots
    W._mem.clear()
    card = _ChainCard(c, waitc)
    card.write(0, img)
    st = RG.run_board_chained(card, progs, at, workers=1)
    segs = RG.chain_segments(words, waitc, c.DRAM_BYTES - at, c.D)
    assert st["runs"] == len(segs) == card.runs == len(card.loads) > 2
    assert [a for a, _, _ in card.loads] == [at] * len(segs)
    assert segs[0][0] == 0 and segs[-1][1] == len(words) and \
        all(a[1] == b[0] for a, b in zip(segs, segs[1:]))
    assert np.array_equal(card.dram[:len(img)], ref)
    small = _ChainCard(dataclasses.replace(g.cfg, DRAM_BYTES=at + one // 2), waitc)
    with pytest.raises(ValueError, match="larger than the room"):
        RG.run_board_chained(small, progs, at, workers=1)
    assert small.loads == [] and small.runs == 0


@pytest.mark.parametrize("small", [False, True], ids=["one-segment", "segments"])
def test_chain_plan_runs_again_on_new_inputs(small):
    """A ChainPlan prepared once runs the graph again after only its input is rewritten: the
    image the sequential runs of a fresh image with that input leave; a single segment's slots
    written once (the runs leave the slot area alone), segments' every run."""
    import dataclasses
    from opentpu import isa as I
    from opentpu import waits as W
    torch.manual_seed(3)
    m = torch.nn.Conv1d(64, 64, 3, 1, 1)
    w, b = m.weight.detach().numpy(), m.bias.detach().numpy()
    g = RG.Graph(cfg())
    xt = g.tensor(16, 64, 2)
    y = g.conv1d(xt, w, b, 1, 1)
    y = g.group_norm(y, 8, np.ones(64), np.zeros(64), 1e-5, act="mish")
    out = g.conv1d(y, w, b, 1, 1)
    progs = g.compile() * 2
    at = -(-(len(g.image()) + (1 << 16)) // 4096) * 4096
    words = [np.asarray(I.assemble(p), np.uint32) for _, p in progs]
    big = max(len(x) for x in words) // 8 + 2
    room = (2 * (-(-32 * big // 128) * 128) + 4096) if small else (1 << 22) - at
    c = dataclasses.replace(g.cfg, DRAM_BYTES=at + room)
    W._mem.clear()
    card = _ChainCard(c, True)
    writes = []
    wr = card.write
    card.write = lambda a, d: (writes.append(a), wr(a, d))
    rng = np.random.default_rng(4)
    img = g.image()
    g.put(img, xt, rng.normal(size=(16, 64)))
    card.write(0, img)
    plan = RG.ChainPlan(card, progs, at, workers=1)
    plan.run(card)
    for k in range(2):
        x = rng.normal(size=(16, 64)).astype(np.float32)
        fresh = g.image()
        g.put(fresh, xt, x)
        ref, _ = RG.run_isa(g.cfg, progs, fresh)
        reg = np.zeros(xt.nbytes, np.uint8)
        g.put(reg, RG.DT(0, xt.L, xt.C, xt.pad, xt.rs), x)
        card.write(xt.addr, reg)
        plan.run(card)
        np.testing.assert_array_equal(g.read(card.dram, out), g.read(ref, out))
        assert np.array_equal(card.dram[:len(fresh)], ref)
    slot_writes = sum(1 for a in writes if a == at)
    assert slot_writes == (3 * plan.stats["runs"] if small else 1)
    assert (plan.stats["runs"] > 1) == small


@pytest.mark.parametrize("tm", [1 << 16, 4096])
def test_group_norm2d_pieces(tm):
    """GroupNorm over [H, W, C] with a residual and SiLU; with a 4096-word TMEM each 24-pixel
    row goes in three pieces of 8 (the SD VAE's 256 x 256 x 256 rows on the card): against
    PyTorch both ways."""
    from dataclasses import replace
    H, W, C, G = 5, 24, 64, 8
    torch.manual_seed(3)
    gn = torch.nn.GroupNorm(G, C, eps=1e-6)
    with torch.no_grad():
        gn.weight.uniform_(0.5, 1.5)
        gn.bias.uniform_(-0.3, 0.3)
    x = torch.randn(1, C, H, W) * 2 + 0.5
    r = torch.randn(1, C, H, W)
    with torch.no_grad():
        want = (torch.nn.functional.silu(gn(x) + r))[0].permute(1, 2, 0).numpy()
    g = RG.Graph(replace(cfg(), TMEM_WORDS=tm))
    xt = g.tensor2(H, W, C, 0, 0)
    rt = g.tensor2(H, W, C, 1, 0)
    y = g.group_norm2d(xt, G, gn.weight.detach().numpy(), gn.bias.detach().numpy(), 1e-6,
                       act="silu", res=rt)
    progs = g.compile()
    img = g.image()
    fl = img.view(np.float32)
    xa = x[0].permute(1, 2, 0).numpy()
    fl[xt.addr // 4:xt.addr // 4 + H * W * C] = xa.reshape(-1)
    for h in range(H):
        o = (rt.addr // 4) + (h + 1) * rt.rs + C
        fl[o:o + W * C] = r[0, :, h].T.reshape(-1).numpy()
    img, _ = RG.run_isa(g.cfg, progs, img)
    fl = img.view(np.float32)
    got = np.stack([fl[y.addr // 4 + (h + y.border) * y.rs + y.border * C:][:W * C]
                    for h in range(H)]).reshape(H, W, C)
    assert rel(got, want) < 1e-4, rel(got, want)


def test_profile_board_and_table():
    """profile_board runs each program on its own and keeps its cycles; profile_table sums
    them by family (numbers and the trailing block index dropped)."""
    class Stub:
        waitc = False

        def __init__(self):
            self.loaded = []

        def load_program(self, at, w, e):
            self.loaded.append(len(w))

        def run(self, timeout):
            return {"cycles": 100 * self.loaded[-1], "instructions": [self.loaded[-1]]}

    words = [np.zeros(n, np.uint32) for n in (1, 2, 3, 4)]
    progs = [("gen.rb01.c1.0", None), ("gen.rb02.c1.1", None), ("stft", None), ("gen.rb11.c1", None)]
    rows = RG.profile_board(Stub(), progs, 0, words=words)
    assert rows == [("gen.rb01.c1.0", 100), ("gen.rb02.c1.1", 200), ("stft", 300),
                    ("gen.rb11.c1", 400)]
    t = RG.profile_table(rows)
    line = next(l for l in t.splitlines() if l.startswith("gen.rb.c "))
    assert line.split()[1] == "0.00" and line.split()[-1] == "3"
    assert t.splitlines()[1].startswith("gen.rb.c")          # the largest family first
