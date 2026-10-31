"""vLLM SamplingParams -> the device sampler (opentpu/vllm/devsample.py), without vLLM."""
from types import SimpleNamespace as P

from opentpu.llm import generate as G
from opentpu.vllm.devsample import device_sampling


def test_greedy_and_penalized_greedy():
    d = device_sampling(P(temperature=0.0, logprobs=None))
    assert d.ok and d.samp is None
    d = device_sampling(P(temperature=0.0, logprobs=None, repetition_penalty=1.3))
    assert d.ok and d.samp.k == 1 and d.samp.penalty == 1.3


def test_sampled_options_map_to_the_device_sampler():
    d = device_sampling(P(temperature=0.7, top_k=40, top_p=0.9, logprobs=None,
                          repetition_penalty=1.1))
    s = d.samp
    assert d.ok and not d.capped and s.k == 40 and s.top_p == 0.9 and s.penalty == 1.1
    assert abs(s.itemp - G.math.log2(G.math.e) / 0.7) < 1e-12
    d = device_sampling(P(temperature=1.0, top_k=-1, top_p=None, logprobs=None))
    assert d.ok and d.capped and d.samp.k == G.K_MAX and d.samp.top_p == 1.0
    d = device_sampling(P(temperature=1.0, top_k=1000, logprobs=None))
    assert d.capped and d.samp.k == G.K_MAX


def test_host_only_options_are_refused():
    base = dict(temperature=0.8, top_k=20, logprobs=None)
    for extra in ({"logprobs": 65}, {"min_p": 0.05}, {"presence_penalty": 0.5},
                  {"frequency_penalty": 0.2}, {"logit_bias": {3: 1.0}},
                  {"allowed_token_ids": [1, 2]}, {"repetition_penalty": 0.9},
                  {"prompt_logprobs": 1}):
        assert not device_sampling(P(**{**base, **extra})).ok, extra


def test_sample_logprobs_are_on_the_device():
    """Sample logprobs (0 .. K_MAX) stay device-sampleable: batch._LogProbs computes them."""
    for t in (0.0, 0.8):
        for n in (0, 5, 64):
            d = device_sampling(P(temperature=t, top_k=20, logprobs=n))
            assert d.ok and d.lp == n
    assert device_sampling(P(temperature=0.0, logprobs=None)).lp is None
