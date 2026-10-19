"""The framework reference for BitNet b1.58 2B4T: HF transformers (torch, CPU, float32).

Usage: python tools/bitnet_hf_reference.py MODEL_DIR IDS.npz OUT.npz [--layers N] [--tokens N]

IDS.npz is a file tools/bitnet_generate.py --save wrote (its `ids`, so both see the same prompt
tokens). With --layers N the model is built with N decoder layers (the checkpoint's later layers
are not loaded), the way bitnet_generate.py --layers N truncates it. Greedy decoding with the KV
cache, no EOS stop, so token i is the argmax after i forwards as in bitnet_generate.py.
OUT.npz holds ids, the first-token logits, the greedy tokens and the library versions. Runs in a
separate environment with torch and transformers (not a dependency of opentpu).
"""
import argparse
import json
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("ids")
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--tokens", type=int, default=8)
    a = ap.parse_args()
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM
    torch.set_grad_enabled(False)
    ids = np.load(a.ids)["ids"].tolist()
    cfg = AutoConfig.from_pretrained(a.model)
    q = getattr(cfg, "quantization_config", None)
    print(f"transformers {transformers.__version__}, torch {torch.__version__}, "
          f"quantization_config {json.dumps(q if isinstance(q, dict) else getattr(q, '__dict__', str(q)), default=str)}")
    if a.layers:
        cfg.num_hidden_layers = a.layers
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model, config=cfg, torch_dtype=torch.float32)
    model.eval()
    lin = sorted({type(m).__name__ for m in model.modules() if "Linear" in type(m).__name__})
    print(f"loaded {cfg.num_hidden_layers} layers in {time.time() - t0:.0f}s, linear classes {lin}")
    cur, past, toks, first = torch.tensor([ids]), None, [], None
    t0 = time.time()
    for i in range(a.tokens):
        o = model(cur, past_key_values=past, use_cache=True)
        past = o.past_key_values
        lg = o.logits[0, -1].float()
        if first is None:
            first = lg.numpy().copy()
        t = int(lg.argmax())
        toks.append(t)
        cur = torch.tensor([[t]])
    print(f"hf greedy {toks} in {time.time() - t0:.1f}s")
    np.savez(a.out, ids=np.asarray(ids, np.int64), logits=first, tokens=np.asarray(toks, np.int64),
             layers=np.int64(cfg.num_hidden_layers), backend=np.array("hf"),
             transformers=np.array(transformers.__version__), torch=np.array(torch.__version__),
             linear_classes=np.array(",".join(lin)))
    print(f"saved {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
