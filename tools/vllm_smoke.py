"""vLLM 0.32.0 on openTPU (opentpu/vllm): a checkpoint through vLLM's LLM engine on the ISA
simulator or the card, its greedy tokens checked against openTPU's own Engine on the same
backend (run after vLLM's engine is shut down: one user of the card at a time).

Usage: OTPU_VLLM_BACKEND=board python tools/vllm_smoke.py MODEL_DIR [--slots 4] [--tokens 16]
           [--max-model-len 512] [--prompts 3]
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
              max_num_seqs=a.slots, additional_config={"opentpu": {"slots": a.slots}})
    print(f"vLLM engine up in {time.time() - t0:.1f}s on {os.environ['OTPU_VLLM_BACKEND']}")
    sp = SamplingParams(temperature=0.0, max_tokens=a.tokens, ignore_eos=True, detokenize=False)
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in ps], sp, use_tqdm=False)
    dt = time.time() - t0
    got = [list(o.outputs[0].token_ids) for o in outs]
    st = llm.collective_rpc(lambda self: dict(self.stats))[0]
    print(f"vLLM: {len(ps)} requests x {a.tokens} tokens in {dt:.2f}s = "
          f"{len(ps) * a.tokens / dt:.2f} tokens/s; worker {st}")
    llm.llm_engine.engine_core.shutdown() if hasattr(llm.llm_engine, "engine_core") else None
    del llm
    gc.collect()
    from opentpu.vllm.worker import build_engine
    ok = True
    for p, g in zip(ps, got):
        e, _ = build_engine(a.model, os.environ["OTPU_VLLM_BACKEND"], a.max_model_len, 1)
        lg, ref = e.prefill(p), []
        for _ in range(a.tokens):
            ref.append(int(np.argmax(lg)))
            if len(ref) < a.tokens:
                lg = e.step(ref[-1])
        del e
        gc.collect()
        m = g == ref
        ok &= m
        print(f"{'MATCH' if m else 'MISMATCH'} prompt[{len(p)}]: vllm {g} engine {ref}")
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
