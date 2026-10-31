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
