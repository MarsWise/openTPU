"""vLLM SamplingParams -> the device sampler (opentpu/llm/generate.py Sampling, the batched
steps' per-row picks, opentpu/llm/batch.py): which requests the card can pick for without any
host arithmetic on logits. No vLLM import (the mapping is tested without it).

The device takes temperature, top-k (1 .. K_MAX), top-p and a repetition penalty >= 1 (Hugging
Face's / vLLM's: positive logits of the context's ids divided by R, negative ones multiplied);
the uniform of each pick is the request's seeded generator's (the host's RNG, as a seed is). An
unset top-k (vLLM's 0 / -1: the whole vocabulary) becomes K_MAX: the candidates are the K_MAX
largest logits, a truncation of the tail the result reports (capped). min-p, presence and
frequency penalties, logit bias, allowed-token lists, bad words and prompt logprobs are not on
the device: such a request is not device-sampleable. Sample logprobs (0 .. K_MAX top ones, vLLM's
raw logprobs) are (llm/batch.py _LogProbs): lp, the count, or None."""
from __future__ import annotations

from types import SimpleNamespace

from ..llm import generate as G


def device_sampling(params):
    """(ok, Sampling or None for greedy, capped) for vLLM SamplingParams `params`: ok False when
    some option needs the host; capped when an unset or larger top-k was cut to K_MAX."""
    no = SimpleNamespace(ok=False, samp=None, capped=False, lp=None)
    lp = getattr(params, "logprobs", None)
    if getattr(params, "prompt_logprobs", None) is not None or \
            (lp is not None and not 0 <= int(lp) <= G.K_MAX):
        return no
    lp = None if lp is None else int(lp)
    for name in ("min_p", "presence_penalty", "frequency_penalty"):
        if float(getattr(params, name, 0.0) or 0.0) != 0.0:
            return no
    for name in ("logit_bias", "allowed_token_ids", "bad_words", "logits_processors"):
        if getattr(params, name, None):
            return no
    rep = float(getattr(params, "repetition_penalty", 1.0) or 1.0)
    if rep < 1.0:
        return no
    t = float(getattr(params, "temperature", 0.0) or 0.0)
    if t < 1e-5:
        return SimpleNamespace(ok=True, capped=False, lp=lp,
                               samp=None if rep == 1.0 else G.Sampling(0.0, 1, 1.0, rep))
    k = int(getattr(params, "top_k", 0) or 0)
    capped = k <= 0 or k > G.K_MAX
    k = G.K_MAX if capped else k
    tp = getattr(params, "top_p", None)
    tp = 1.0 if tp is None else float(tp)
    return SimpleNamespace(ok=True, capped=capped, lp=lp, samp=G.Sampling(t, k, tp, rep))
