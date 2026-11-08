"""Strict vLLM worker guards, with isolated vLLM interfaces and no board."""
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def worker(monkeypatch):
    # The worker is tested directly; vLLM is an optional Linux dependency.
    def module(name, **attrs):
        value = types.ModuleType(name)
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    for name in ("vllm", "vllm.v1", "vllm.v1.worker"):
        module(name, __path__=[])
    module("vllm.logger", init_logger=lambda _: NS(info=lambda *a, **k: None))
    module("vllm.v1.kv_cache_interface", FullAttentionSpec=object)
    module("vllm.v1.outputs", LogprobsLists=object, ModelRunnerOutput=object)
    module("vllm.v1.worker.worker_base", CompilationTimes=object, WorkerBase=object)
    module("torch")
    class Unsupported(Exception):
        pass
    module("opentpu.vllm.platform", BLOCK_SIZE=16, Unsupported=Unsupported)
    path = Path(__file__).resolve().parents[1] / "opentpu" / "vllm" / "worker.py"
    spec = importlib.util.spec_from_file_location("opentpu.vllm._worker_rng_test", path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


def fake_worker(module, vrand):
    w = object.__new__(module.OpenTPUWorker)
    w.free, w.kept, w.opts = [0, 1], {}, {"fpga_only": True}
    w.eng = NS(cfg=NS(VRAND=vrand), reset=lambda **kw: None, poss=[0, 0])
    w.steps = NS(set_sampling=lambda *a, **kw: None)
    w.stats, w.reqs = {}, {}
    w.B = 2
    return w


def test_strict_sampled_request_requires_device_rng_before_reserving_slot(worker):
    w = fake_worker(worker, False)
    params = NS(temperature=.8, top_k=20, logprobs=None)
    with pytest.raises(worker.Unsupported, match="RNG|VRAND"):
        w._take_slot("req", [1], params, 7)
    assert w.free == [0, 1] and not w.reqs


def test_rejected_resumed_request_keeps_preemption_state(worker):
    w = fake_worker(worker, False)
    params = NS(temperature=.8, top_k=20, logprobs=None)
    state = (params, object())
    w.kept["req"] = state
    with pytest.raises(worker.Unsupported, match="VRAND"):
        w._take_slot("req", [1], None, None)
    assert w.kept["req"] is state and w.free == [0, 1]


def test_strict_sampled_request_uses_device_rng_when_available(worker):
    w = fake_worker(worker, True)
    params = NS(temperature=.8, top_k=20, logprobs=None)
    calls = []
    w.steps.set_sampling = lambda *args, **kw: calls.append((args, kw))
    req = w._take_slot("req", [1], params, 7)
    assert req.rid == "req" and len(calls) == 1
    assert 0 <= calls[0][1]["seed"] < 2**32


def test_non_strict_sampled_request_without_device_rng_is_still_allowed(worker):
    w = fake_worker(worker, False)
    w.opts["fpga_only"] = False
    params = NS(temperature=.8, top_k=20, logprobs=None)
    assert w._take_slot("req", [1], params, 7).rid == "req"


def test_strict_greedy_without_rng_never_draws_host_uniform(worker):
    w = fake_worker(worker, False)
    params = NS(temperature=0., logprobs=None)
    req = w._take_slot("req", [1], params, 7)
    class ForbiddenRng:
        def random(self):
            pytest.fail("strict greedy drew a host uniform")
    req.rng = ForbiddenRng()
    w.model_config = NS(seed=0)
    w.stats = {"prefill_runs": 0, "decode_runs": 0, "decode_rows": 0,
               "device_s": 0, "steps": 0}
    w.steps.runs = 0
    w.steps.timing = dict(cycles=0, programs=0, reused=0, compile_wait_s=0.)
    w.steps.run = lambda rows, tokens, uniforms: [4]
    scheduled = NS(finished_req_ids=[], preempted_req_ids=[], scheduled_new_reqs=[],
                   scheduled_cached_reqs=NS(req_ids=[], resumed_req_ids=[]),
                   num_scheduled_tokens={"req": 1})
    w.execute_model(scheduled)
    assert w._pending[1]["req"] == 4


def test_strict_decode_refuses_sampling_if_rng_capability_disappears(worker):
    w = fake_worker(worker, True)
    params = NS(temperature=.8, top_k=20, logprobs=None)
    req = w._take_slot("req", [1], params, 7)
    class ForbiddenRng:
        def random(self):
            pytest.fail("strict sampling drew a host uniform")
    req.rng = ForbiddenRng()
    w.eng.cfg.VRAND = False
    w.model_config = NS(seed=0)
    w.stats = {"prefill_runs": 0, "decode_runs": 0, "decode_rows": 0,
               "device_s": 0, "steps": 0}
    w.steps.runs = 0
    scheduled = NS(finished_req_ids=[], preempted_req_ids=[], scheduled_new_reqs=[],
                   scheduled_cached_reqs=NS(req_ids=[], resumed_req_ids=[]),
                   num_scheduled_tokens={"req": 1})
    with pytest.raises(worker.Unsupported, match="VRAND"):
        w.execute_model(scheduled)
