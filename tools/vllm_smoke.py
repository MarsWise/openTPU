"""vLLM 0.32.0 on openTPU (opentpu/vllm): a checkpoint through vLLM's LLM engine on the ISA
simulator or the card, its greedy tokens checked against openTPU's own Engine on the same
backend (run after vLLM's engine is shut down: one user of the card at a time).

Usage: OTPU_VLLM_BACKEND=board python tools/vllm_smoke.py MODEL_DIR [--slots 4] [--tokens 16]
           [--max-model-len 512] [--prompts 3] [--head-format int4]
Needs vLLM 0.32.0 in the environment (docs/nextgen/VLLM_INTEGRATION.md). Prints per-request
tokens, vLLM's wall time and tokens/s, and MATCH / MISMATCH per prompt.
"""
import argparse
import gc
import os
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--prompts", type=int, default=3)
    ap.add_argument("--head-format", default=None, choices=["int8", "fp4", "int4", "nvfp4"],
                    help="the LM head's format (default: the model's)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="> 0: sampled requests (seeded), picked on the device (OTPU_VLLM_FPGA_ONLY "
                         "set: nothing sampled on the host), checked against the decode loop's "
                         "sampler (generate.reference_pick) on the engine's logits with the "
                         "requests' uniforms")
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--logprobs", type=int, default=None,
                    help="request sample logprobs (computed on the device, batch._LogProbs) and "
                         "check them against the reference engine's log-softmax")
    a = ap.parse_args()
    os.environ.setdefault("OTPU_VLLM_BACKEND", "isa")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    rng = np.random.default_rng(0)
    from opentpu.llm import load_spec
    V = load_spec(a.model).vocab
    ps = [[int(t) for t in rng.integers(100, min(V, 30000), n)] for n in
          ([7, 23, 12, 40, 5, 17, 9, 31][:a.prompts])]
    t0 = time.time()
    llm = LLM(model=a.model, skip_tokenizer_init=True, max_model_len=a.max_model_len,
              max_num_seqs=a.slots, additional_config={"opentpu": {"slots": a.slots,
                                                "head_format": a.head_format}})
    print(f"vLLM engine up in {time.time() - t0:.1f}s on {os.environ['OTPU_VLLM_BACKEND']}")
    if a.temperature > 0:
        os.environ["OTPU_VLLM_FPGA_ONLY"] = "1"
    sps = [SamplingParams(temperature=a.temperature, top_k=a.top_k, top_p=a.top_p,
                          repetition_penalty=a.repetition_penalty, seed=1000 + i,
                          max_tokens=a.tokens, ignore_eos=True, detokenize=False,
                          logprobs=a.logprobs)
           if a.temperature > 0 else
           SamplingParams(temperature=0.0, max_tokens=a.tokens, ignore_eos=True,
                          detokenize=False, logprobs=a.logprobs) for i in range(len(ps))]
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in ps], sps, use_tqdm=False)
    dt = time.time() - t0
    got = [list(o.outputs[0].token_ids) for o in outs]
    glp = [o.outputs[0].logprobs for o in outs]
    st = llm.collective_rpc(lambda self: dict(self.stats))[0]
    print(f"vLLM: {len(ps)} requests x {a.tokens} tokens in {dt:.2f}s = "
          f"{len(ps) * a.tokens / dt:.2f} tokens/s; worker {st}")
    llm.llm_engine.engine_core.shutdown() if hasattr(llm.llm_engine, "engine_core") else None
    del llm
    gc.collect()
    from opentpu.vllm.worker import build_engine
    ok = True
    # one reference engine (one KV slot), reset between the prompts and closed at the end: a
    # second Engine on the card would find the first still holding /dev/xdma0 (DeviceBusy)
    e, _ = build_engine(a.model, os.environ["OTPU_VLLM_BACKEND"], a.max_model_len, 1,
                        head_format=a.head_format)
    try:
        from opentpu.llm import generate as G
        from opentpu.vllm.devsample import device_sampling
        for i, (p, g) in enumerate(zip(ps, got)):
            e.reset()
            lg, ref = e.prefill(p), []
            samp = device_sampling(sps[i]).samp
            rng = np.random.default_rng(1000 + i)
            lp_err, lp_ok = 0.0, True
            for j in range(a.tokens):
                if samp is None:
                    ref.append(int(np.argmax(lg)))
                else:
                    u = np.minimum(np.float32(rng.random()), np.float32(1 - 2.0 ** -24))
                    ref.append(int(G.reference_pick(np.asarray(lg), samp, p + ref, u)))
                if a.logprobs is not None and glp[i] is not None and j < len(glp[i]) \
                        and j < len(g) and g[j] == ref[-1]:
                    x = np.asarray(lg, np.float64)
                    lp = x - (x.max() + np.log(np.exp(x - x.max()).sum()))
                    d = glp[i][j]
                    want = [int(t) for t in np.argsort(-lp, kind="stable")[:a.logprobs]]
                    lp_ok &= all(t in d for t in want) and d[ref[-1]].rank == \
                        int((lp > lp[ref[-1]]).sum()) + 1
                    lp_err = max([lp_err] + [abs(d[t].logprob - lp[t]) for t in d])
                if len(ref) < a.tokens:
                    lg = e.step(ref[-1])
            m = g == ref
            if a.logprobs is not None:
                m = m and glp[i] is not None and lp_ok and lp_err < 2e-2
                print(f"logprobs prompt[{len(p)}]: top {a.logprobs} ids and ranks "
                      f"{'equal' if lp_ok else 'DIFFER'}, max |d logprob| {lp_err:.2e}")
            ok &= m
            print(f"{'MATCH' if m else 'MISMATCH'} prompt[{len(p)}]: vllm {g} engine {ref}")
    finally:
        close = getattr(e.backend, "close", None)
        if close is not None:
            close()
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
