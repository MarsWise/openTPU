"""Integrate lifetime planning with real CNN layout and execution."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from opentpu import Config, cnngraph as G


def test_compact_cnn_matches_original_isa_layout():
    torch.manual_seed(17)
    model = torch.nn.Sequential(torch.nn.MaxPool2d(1, 1),
                                torch.nn.MaxPool2d(1, 1),
                                torch.nn.MaxPool2d(1, 1)).eval()
    graph = torch.fx.symbolic_trace(model)
    cfg = Config(D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16,
                 TMEM_WORDS=65536, DRAM_BYTES=1 << 24)
    x = np.random.default_rng(12).standard_normal((4, 4, 3)).astype(np.float32)
    plans = [G.build(graph, (3, 4, 4), cfg, reuse_memory=reuse) for reuse in (False, True)]
    outputs = []
    for plan in plans:
        image = plan.image(plan.size, cfg.D)
        plan.put_input(image, 'input_1', x)
        outputs.append(plan.read(plan.run_isa(cfg, image), plan.env[plan.out_node]))
    np.testing.assert_array_equal(outputs[0], outputs[1])
    np.testing.assert_array_equal(outputs[1], x)
    assert plans[1].size < plans[0].size
    plans[1].memory_plan.verify()


def test_opaque_kernel_disables_buffer_reuse():
    plan = G.Plan()
    a, b = plan.buf('a', 4, 4, 3), plan.buf('b', 4, 4, 3)
    plan.inputs['x'] = a
    plan.ops.append(G.Op('copy', 'copy', G.Val(a), G.Val(b)))
    plan.ops.append(G.Op('kernel', 'opaque', None, G.Val(b)))
    plan.out_node = 'out'
    plan.env = {'out': G.Val(b)}
    plan.layout(32, reuse_memory=True)
    assert a.addr + a.nbytes <= b.addr or b.addr + b.nbytes <= a.addr


def test_borders_and_tails_are_not_reused():
    plan = G.Plan()
    a, b, c = (plan.buf(name, 4, 4, 3) for name in ('a', 'b', 'c'))
    plan.inputs['x'] = a
    plan.ops.extend((G.Op('pool', 'padded', G.Val(a), G.Val(b), {'pad': 1}),
                     G.Op('copy', 'copy', G.Val(b), G.Val(c))))
    plan.out_node = 'out'
    plan.env = {'out': G.Val(c)}
    plan.layout(32, reuse_memory=True)
    assert a.border == 1
    assert all(a.addr + a.nbytes <= other.addr or other.addr + other.nbytes <= a.addr
               for other in (b, c))


def test_residual_cnn_matches_non_reused_layout():
    class Residual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Conv2d(3, 4, 3, padding=1)
            self.b = torch.nn.Conv2d(4, 4, 1)

        def forward(self, x):
            y = self.a(x)
            return self.b(y) + y

    torch.manual_seed(42)
    graph = torch.fx.symbolic_trace(Residual().eval())
    cfg = Config(D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16,
                 TMEM_WORDS=65536, DRAM_BYTES=1 << 24)
    x = np.random.default_rng(6).standard_normal((4, 4, 3)).astype(np.float32)
    outputs = []
    for reuse in (False, True):
        plan = G.build(graph, (3, 4, 4), cfg, reuse_memory=reuse)
        image = plan.image(plan.size, cfg.D)
        plan.put_input(image, 'x', x)
        outputs.append(plan.read(plan.run_isa(cfg, image), plan.env[plan.out_node]))
    np.testing.assert_array_equal(outputs[0], outputs[1])
