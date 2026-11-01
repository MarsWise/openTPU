"""Activation steering on the device (kernels/steering.py, Engine(steering=True)): with every
word 0 the logits are the unsteered engine's bit for bit; steered, the device follows the
quantized math's steering (emulated_logits(steer=)); a scale changes with a DRAM write, no
recompilation; the resident decode loop on the card steers too."""
import numpy as np
import pytest

pytest.importorskip("transformers")
pytest.importorskip("torch")

from opentpu.kernels import steering as ST  # noqa: E402
from opentpu.llm import qwen3 as Q3  # noqa: E402
from test_autodecode import _tiny  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return _tiny("qwen3")


def _cos(a, b):
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


def _vectors(spec, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(spec.layers, spec.hidden)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_zero_words_are_the_unsteered_model(model):
    W, spec = model
    toks = [5, 77, 300, 12, 999]
    a = Q3.Engine(spec, W, cap=512)
    b = Q3.Engine(spec, W, cap=512, steering=True)
    assert b.set_steering(_vectors(spec)) is None          # (all words 0: no profile key)
    for t in toks:
        la, lb = a.step(t), b.step(t)
        assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    with pytest.raises(ValueError, match="no steering"):
        a.set_steering(_vectors(spec), s_mlp=1.0)


def test_steered_logits_follow_the_quantized_math(model):
    W, spec = model
    toks = [5, 77, 300, 12, 999, 3]
    v = _vectors(spec, 1)
    w = ST.words(s_attn=2.0, p_attn=-0.5, s_mlp=3.0, p_mlp=0.25)
    eng = Q3.Engine(spec, W, cap=512, steering=True)
    key = eng.set_steering(v * 1.0, *w)
    assert key is not None and key.startswith(ST.profile_hash(v))
    dev = np.array([eng.step(t) for t in toks], np.float64)
    emu = Q3.emulated_logits(spec, W, toks, steer=(v, w))
    plain = Q3.emulated_logits(spec, W, toks)
    assert _cos(dev, emu).min() > 0.999
    assert _cos(dev, plain).min() < 0.999            # (the steering moved the logits)


def test_a_new_scale_needs_no_recompilation(model):
    """The step's program is the same whatever the words (they are DRAM data the program
    loads); a new scale written into the DRAM moves the logits as the math says."""
    from opentpu import isa as I
    W, spec = model
    v = _vectors(spec, 2)
    eng = Q3.Engine(spec, W, cap=512, steering=True)
    eng.set_steering(v, s_mlp=1.5)
    p1 = [I.assemble(p) for p in eng.image.compile_step(0)]
    l1 = eng.step(5)
    eng2 = Q3.Engine(spec, W, cap=512, steering=True)
    eng2.set_steering(v, s_mlp=4.0)
    p2 = [I.assemble(p) for p in eng2.image.compile_step(0)]
    l2 = eng2.step(5)
    assert all(np.array_equal(a, b) for a, b in zip(p1, p2))
    assert not np.array_equal(l1, l2)
    emu = Q3.emulated_logits(spec, W, [5], steer=(v, ST.words(s_mlp=4.0)))
    assert _cos(l2[None].astype(np.float64), emu).min() > 0.999


def test_the_card_loop_steers(model):
    """The resident decode loop (generate_card) with the steering words: its greedy tokens are
    the host loop's over steered steps."""
    W, spec = model
    v = _vectors(spec, 3)
    w = dict(s_attn=1.0, s_mlp=2.5)
    a = Q3.Engine(spec, W, cap=512, resident=True, steering=True)
    b = Q3.Engine(spec, W, cap=512, steering=True)
    for e in (a, b):
        e.set_steering(v, **w)
    toks = [5, 77, 300, 12]
    la, lb = a.prefill(toks), b.prefill(toks)
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    t0, ref, t = int(np.argmax(la)), [], int(np.argmax(la))
    for _ in range(6):
        t = int(np.argmax(b.step(t)))
        ref.append(t)
    assert a.generate_card(t0, 6, stop_ids=[]) == ref


def test_a_new_profile_restarts_the_chat_context(model):
    """otpu-chat's context fed under one steering key is not reused under another: the next
    turn after set_steering feeds the whole conversation again (reused 0)."""
    from opentpu.host.chat import Chat, sampler
    from test_autodecode import _Tok
    W, spec = model
    eng = Q3.Engine(spec, W, cap=512, resident=True, steering=True)
    chat = Chat(eng, _Tok(), False, sampler(0, 0, 1.0, None), 4)
    chat.ask("hello there")
    _, t2 = chat.ask("and again")
    assert t2.reused > 0
    eng.set_steering(_vectors(spec, 4), s_mlp=2.0)
    _, t3 = chat.ask("steered now")
    assert t3.reused == 0
    _, t4 = chat.ask("same profile")
    assert t4.reused > 0
