"""openTPU as a vLLM platform (opentpu/vllm, pinned vLLM 0.32.0) on the ISA simulator: vLLM's
own engine (scheduler, continuous batching, request lifecycle) with OpenTPUWorker running every
token on openTPU's engine. A tiny random Qwen3 checkpoint (transformers' save_pretrained):

- greedy tokens through vLLM equal a fresh Engine's for each prompt, with more requests than KV
  slots (they wait for one) and their decode steps batched over the slots;
- a seeded sampled request gives the same tokens alone and among others;
- a preempted request (too few KV blocks) is recomputed and its tokens do not change;
- an aborted request frees its slot; a prompt past max_model_len and an unsupported sampling
  feature (logprobs) are refused, nothing computed for them on the CPU;
- the platform activates only with OTPU_VLLM_BACKEND set.
Skipped without vLLM 0.32.0 (an isolated environment: docs/nextgen/VLLM_INTEGRATION.md)."""
import os

import numpy as np
import pytest

vllm = pytest.importorskip("vllm")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from opentpu import vllm as OV                           # noqa: E402

if vllm.__version__.split("+")[0] != OV.VLLM_VERSION:
    pytest.skip(f"vLLM {vllm.__version__}, the plugin is pinned to {OV.VLLM_VERSION}",
                allow_module_level=True)

MAXLEN = 256
SLOTS = 2


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny_qwen3")
    torch.manual_seed(0)
    hc = transformers.Qwen3Config(
        hidden_size=256, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=128, intermediate_size=512, vocab_size=1000, rms_norm_eps=1e-6,
        rope_theta=1e6, tie_word_embeddings=True, max_position_embeddings=4096,
        initializer_range=0.2)
    transformers.Qwen3ForCausalLM(hc).float().eval().save_pretrained(d)
    # a word-level tokenizer ("w5 w9" -> [5, 9]) for the OpenAI server's text
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    vocab = dict({f"w{i}": i for i in range(1, 1000)}, **{"<unk>": 0})
    t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer, t.decoder = pre_tokenizers.WhitespaceSplit(), decoders.WordPiece()
    t.save(str(d / "tokenizer.json"))
    (d / "tokenizer_config.json").write_text(
        '{"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<unk>", '
        '"eos_token": "w999", "model_max_length": 256}')
    return str(d)


@pytest.fixture(scope="module")
def env():
    old = {k: os.environ.get(k) for k in (OV.ENV_BACKEND, "VLLM_ENABLE_V1_MULTIPROCESSING")}
    os.environ[OV.ENV_BACKEND] = "isa"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    yield
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _llm(ckpt, **kw):
    from vllm import LLM
    return LLM(model=ckpt, skip_tokenizer_init=True, max_model_len=MAXLEN, dtype="float32",
               max_num_seqs=SLOTS, additional_config={"opentpu": {"slots": SLOTS}}, **kw)


@pytest.fixture(scope="module")
def llm(ckpt, env):
    return _llm(ckpt)


def _prompts(seed, sizes):
    r = np.random.default_rng(seed)
    return [[int(x) for x in r.integers(0, 1000, n)] for n in sizes]


def _ref(ckpt, prompt, n):
    """Greedy tokens of a fresh one-slot engine (Engine.prefill, then step)."""
    from opentpu.vllm.worker import build_engine
    e, _ = build_engine(ckpt, "isa", MAXLEN, 1)
    lg, out = e.prefill(prompt), []
    for _ in range(n):
        out.append(int(np.argmax(lg)))
        if len(out) < n:
            lg = e.step(out[-1])
    return out


def _gen(llm, prompts, **sp):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    sp = dict(dict(temperature=0.0, ignore_eos=True, detokenize=False), **sp)
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                        [SamplingParams(**dict(sp, **({"max_tokens": m} if isinstance(m, int)
                                                      else m)))
                         for m in sp.pop("each")] if "each" in sp else SamplingParams(**sp),
                        use_tqdm=False)
    return [list(o.outputs[0].token_ids) for o in outs], [o.outputs[0].finish_reason
                                                          for o in outs]


def _stats(llm):
    return llm.collective_rpc(lambda self: dict(self.stats))[0]


def test_platform_activates_only_when_asked(monkeypatch):
    monkeypatch.delenv(OV.ENV_BACKEND, raising=False)
    assert OV.platform_plugin() is None
    monkeypatch.setenv(OV.ENV_BACKEND, "isa")
    assert OV.platform_plugin() == "opentpu.vllm.platform.OpenTPUPlatform"
    monkeypatch.setenv(OV.ENV_BACKEND, "cuda")
    with pytest.raises(ValueError):
        OV.platform_plugin()


def test_greedy_tokens_are_the_engines_with_requests_batched(llm, ckpt):
    from vllm.platforms import current_platform
    assert current_platform.device_name == "opentpu" and not current_platform.is_cuda()
    ps = _prompts(1, [5, 17, 3, 40, 9])               # five requests, two slots
    n = [7, 4, 9, 5, 6]
    s0 = _stats(llm)
    got, fin = _gen(llm, ps, each=n)
    assert fin == ["length"] * len(ps)
    for p, k, g in zip(ps, n, got):
        assert g == _ref(ckpt, p, k)
    s1 = _stats(llm)
    rows, runs = s1["decode_rows"] - s0["decode_rows"], s1["decode_runs"] - s0["decode_runs"]
    assert runs > 0 and rows > runs                    # decode steps batched over the slots


def test_seeded_sampling_alone_and_among_others(llm):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    p = _prompts(2, [11])[0]
    sp = SamplingParams(temperature=0.8, top_k=50, top_p=0.9, seed=1234, max_tokens=8,
                        ignore_eos=True, detokenize=False)
    alone = llm.generate([TokensPrompt(prompt_token_ids=p)], sp, use_tqdm=False)
    others = _prompts(3, [6, 13])
    mixed = llm.generate([TokensPrompt(prompt_token_ids=q) for q in [others[0], p, others[1]]],
                         [SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True,
                                         detokenize=False), sp,
                          SamplingParams(temperature=1.0, seed=7, max_tokens=8,
                                         ignore_eos=True, detokenize=False)], use_tqdm=False)
    a, b = list(alone[0].outputs[0].token_ids), list(mixed[1].outputs[0].token_ids)
    assert a == b and len(a) == 8


def test_logprobs_are_the_device_logits(llm, ckpt):
    """Sample logprobs (vLLM's raw_logprobs: log-softmax of the device's logits): the sampled
    token's and the top 3 equal those of the Engine's logits at each step."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from opentpu.vllm.worker import build_engine
    p = _prompts(8, [9])[0]
    o = llm.generate([TokensPrompt(prompt_token_ids=p)],
                     SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True,
                                    detokenize=False, logprobs=3), use_tqdm=False)[0].outputs[0]
    e, _ = build_engine(ckpt, "isa", MAXLEN, 1)
    lg = e.prefill(p)
    for t, d in zip(o.token_ids, o.logprobs):
        x = lg.astype(np.float64)
        lp = x - (x.max() + np.log(np.exp(x - x.max()).sum()))
        assert t == int(np.argmax(lg)) and d[t].rank == 1
        for tid, v in d.items():
            assert abs(v.logprob - lp[tid]) < 1e-5
        assert set(d) == {t} | set(np.argsort(-lp)[:3].tolist())
        lg = e.step(t)


def test_preempted_requests_are_recomputed(ckpt, env, llm):
    """KV blocks for one request of max_model_len (num_gpu_blocks_override 17: 16 blocks of 16
    and vLLM's null block): two requests of 144 positions each do not fit together, the
    scheduler preempts one (recompute), and the tokens are still the engine's."""
    small = _llm(ckpt, num_gpu_blocks_override=17)
    ps = _prompts(4, [30, 34])
    got, _ = _gen(small, ps, max_tokens=110)
    st = small.collective_rpc(lambda self: dict(self.stats, resumed=self.resumed))[0]
    for p, g in zip(ps, got):
        assert g == _ref(ckpt, p, 110)
    assert st["resumed"] > 0, st
    del small


def test_bitnet_through_vllm(env, tmp_path_factory):
    """BitNet b1.58 (packed ternary checkpoint, quantization_config "bitnet"): vLLM accepts it
    through the platform's registered stand-ins and the worker runs it as TMM's ternary image;
    greedy tokens == the engine's."""
    import sys
    from pathlib import Path
    from vllm import LLM
    sys.path.insert(0, str(Path(__file__).parent))
    from test_bitnet_engine import make_ckpt

    from opentpu.vllm.worker import build_engine
    d = make_ckpt(tmp_path_factory.mktemp("bitnet_vllm"))
    llm = LLM(model=str(d), skip_tokenizer_init=True, max_model_len=128, max_num_seqs=1)
    p = [1, 5, 9, 33, 64]
    got, _ = _gen(llm, [p], max_tokens=6)
    e, spec = build_engine(d, "isa", 128, 1)
    assert spec.subln and e.image.wformat == "ternary"
    lg, ref = e.prefill(p), []
    for _ in range(6):
        ref.append(int(np.argmax(lg)))
        lg = e.step(ref[-1])
    assert got[0] == ref
    del llm


def test_abort_frees_the_slot_and_refusals(llm, ckpt):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    eng = llm.llm_engine
    sp = SamplingParams(temperature=0.0, max_tokens=50, ignore_eos=True, detokenize=False)
    ps = _prompts(5, [8, 8])
    for i, p in enumerate(ps):
        eng.add_request(f"ab{i}", TokensPrompt(prompt_token_ids=p), sp)
    for _ in range(4):
        eng.step()
    eng.abort_request(["ab0", "ab1"])
    while eng.has_unfinished_requests():
        eng.step()
    q = _prompts(6, [12])[0]             # (vLLM tells the worker of the aborts with its next step)
    assert _gen(llm, [q], max_tokens=5)[0][0] == _ref(ckpt, q, 5)
    assert llm.collective_rpc(lambda self: (len(self.reqs), len(self.free)))[0] == (1, SLOTS - 1)
    _gen(llm, [q], max_tokens=1)
    assert llm.collective_rpc(lambda self: (len(self.reqs), len(self.free)))[0] == (1, SLOTS - 1)
    with pytest.raises(Exception, match="penalty"):
        _gen(llm, [q], max_tokens=2, presence_penalty=0.5)
    with pytest.raises(Exception):                     # vLLM's own length check
        _gen(llm, [list(range(MAXLEN + 4))], max_tokens=2)


def test_openai_server_streams_and_refuses(ckpt, tmp_path):
    """`vllm serve` (its engine core in its own process, the worker there): a streamed
    completion's tokens are the engine's greedy tokens; an unsupported feature is a 400 and the
    server goes on."""
    import json
    import socket
    import subprocess
    import sys
    import time
    import urllib.error
    import urllib.request
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(os.environ, **{OV.ENV_BACKEND: "isa"})
    env.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
    log = open(tmp_path / "serve.log", "w")
    proc = subprocess.Popen([sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", ckpt,
                             "--max-model-len", str(MAXLEN), "--max-num-seqs", str(SLOTS),
                             "--dtype", "float32", "--port", str(port), "--served-model-name",
                             "tiny"], env=env, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"

    def post(body):
        rq = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(),
                                    {"Content-Type": "application/json"})
        return urllib.request.urlopen(rq, timeout=120)
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(url + "/health", timeout=2)
                break
            except (urllib.error.URLError, ConnectionError):
                assert proc.poll() is None, (tmp_path / "serve.log").read_text()[-3000:]
                time.sleep(0.5)
        p = _prompts(7, [6])[0]
        r = post({"model": "tiny", "prompt": " ".join(f"w{t}" for t in p), "max_tokens": 5,
                  "temperature": 0, "ignore_eos": True, "stream": True})
        text = ""
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                text += json.loads(line[6:])["choices"][0]["text"]
        want = _ref(ckpt, [t if t else 0 for t in p], 5)
        assert [int(w[1:]) for w in text.split()] == want
        with pytest.raises(urllib.error.HTTPError) as e:
            post({"model": "tiny", "prompt": "w1 w2", "max_tokens": 2, "presence_penalty": 0.5})
        assert e.value.code == 400 and "penalty" in e.value.read().decode()
        r = json.loads(post({"model": "tiny", "prompt": "w1 w2", "max_tokens": 2}).read())
        assert r["usage"]["completion_tokens"] == 2
    finally:
        proc.terminate()
        proc.wait(60)
