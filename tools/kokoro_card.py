"""Kokoro-82M v1.0 text to speech with every model operation on the device (opentpu/kokoro_graph.py):
graph A (PL-BERT, the duration and text encoders, the durations), then graph B (alignment,
F0 / N, decoder, SineGen with VOP RAND noise, STFT, generator, iSTFT) -> a 24 kHz WAV.

    python tools/kokoro_card.py --weights models/kokoro --config config.json [--isa]
        [--phonemes "həlˈO wˈɜɹld"] [--voice af_heart] [--seed 1234] [--out kokoro.wav] [--ref]

The host: phonemes -> ids, the voice row, the durations read and rounded (graph B's shape),
the seed. --ref: the float reference (tools/kokoro/ref, hexgrad's modules, on the host CPU: a
check, not part of the run) given the device's noise: the stages' cosines, the waveform's and
the log-mel spectrogram's (the waveform's phase follows F0 closely: a few Hz apart shift it,
so the log-mel is the audio's measure). Nothing is loaded into the FPGA. Prints CHECK lines."""
from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def logmel(a, n_fft=1024, hop=256, n_mels=80, sr=24000):
    a = np.asarray(a, np.float64)
    w = np.hanning(n_fft + 1)[:-1]
    fr = np.stack([a[i:i + n_fft] * w for i in range(0, len(a) - n_fft, hop)])
    S = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    mel = lambda h: 2595 * np.log10(1 + h / 700)                 # noqa: E731
    pts = 700 * (10 ** (np.linspace(mel(0), mel(sr / 2), n_mels + 2) / 2595) - 1)
    fb = np.zeros((n_mels, len(f)))
    for k in range(n_mels):
        lo, c, hi = pts[k], pts[k + 1], pts[k + 2]
        fb[k] = np.clip(np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)), 0, None)
    return np.log(S @ fb.T + 1e-5)


def cos(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def write_wav(path, x, sr=24000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(np.asarray(x), -1, 1) * 32767).astype("<i2").tobytes())


class IsaRunner:
    def __init__(self, cfg):
        self.cfg, self.img = cfg, None

    def write(self, img):
        self.img = img

    def write_at(self, at, a):
        v = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
        if at + v.size > len(self.img):
            self.img = np.concatenate([self.img, np.zeros(at + v.size - len(self.img), np.uint8)])
        self.img[at:at + v.size] = v

    def run(self, progs, top):
        from opentpu import resident as RG
        if len(self.img) < top:
            self.img = np.concatenate([self.img, np.zeros(top - len(self.img), np.uint8)])
        t0 = time.time()
        self.img, n = RG.run_isa(self.cfg, progs, self.img)
        return {"instructions": n, "run_s": round(time.time() - t0, 1)}

    def read(self, g, t):
        return g.read(self.img, t)


class BoardRunner:
    def __init__(self, board):
        self.b = board

    def write(self, img):
        self.b.write(0, img)

    def write_at(self, at, a):
        self.b.write(at, np.ascontiguousarray(a).view(np.uint8).reshape(-1))

    def run(self, progs, top):
        from opentpu import resident as RG
        at = -(-(top + (1 << 20)) // 4096) * 4096
        return RG.run_board_chained(self.b, progs, at)

    def read(self, g, t):
        n = t.nbytes
        raw = self.b.read(t.addr, n).view(np.float32)
        rs = t.rs
        return np.stack([raw[(t.pad + r) * rs:(t.pad + r) * rs + t.C] for r in range(t.L)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", default=str(ROOT / "models/kokoro"),
                    help="tools/kokoro/convert.py's output: kokoro-v1_0.safetensors, <voice>.safetensors")
    ap.add_argument("--config", required=True, help="the checkpoint's config.json (the vocabulary)")
    ap.add_argument("--phonemes", default="həlˈO wˈɜɹld")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="kokoro.wav")
    ap.add_argument("--isa", action="store_true")
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--ref", action="store_true", help="compare with the float reference")
    a = ap.parse_args()
    from safetensors.numpy import load_file
    from opentpu import kokoro_graph as KG
    from opentpu import resident as RG
    fails = 0

    def check(ok, what, detail=""):
        nonlocal fails
        print(f"CHECK {'PASS' if ok else 'FAIL'}: {what}" + (f" ({detail})" if detail else ""),
              flush=True)
        fails += 0 if ok else 1

    cfgj = json.loads(Path(a.config).read_text(encoding="utf-8"))
    W = load_file(str(Path(a.weights) / "kokoro-v1_0.safetensors"))
    pack = load_file(str(Path(a.weights) / f"{a.voice}.safetensors"))[a.voice]
    ids = KG.ids_of(cfgj["vocab"], a.phonemes)
    style = pack[len(ids) - 3].reshape(-1)              # KPipeline: the row for the phonemes
    ecc0 = None
    if a.isa:
        from opentpu.isasim import board_config
        cfg = board_config(MCOLS=8, TFN=True, VCMP=True, VRAND=True, TMEM_WORDS=131072,
                           DRAM_BYTES=1 << 31)
        run = IsaRunner(cfg)
    else:
        from opentpu.host import memcal
        from opentpu.host.board import Board, XdmaTransport, device_config
        board = Board(XdmaTransport(a.dev), check=False)
        info = board.info()
        cfg = device_config(info)
        for need in ("TFN", "VCMP", "VRAND"):
            if not getattr(cfg, need, False):
                raise SystemExit(f"the bitstream has no {need}")
        ecc0 = memcal.ecc(board.t)
        print(f"card: build {info.get('build_id') or 0:08x}, ECC {ecc0}", flush=True)
        run = BoardRunner(board)
    g = RG.Graph(cfg)
    kk = KG.Kokoro(W, cfgj)
    t0 = time.time()
    dur = kk.build_a(g, ids, style)
    na = len(g.ops)
    pa = g.compile()
    img = g.image(-(-g.top // 4096) * 4096)
    g.put(img, kk.ref_s, style)
    print(f"graph A: {len(ids)} ids, {len(pa)} programs, compiled in {time.time() - t0:.1f} s",
          flush=True)
    t_all = time.time()
    run.write(img)
    sa = run.run(pa, g.top)
    frames = KG.round_durations(run.read(g, dur))
    print(f"graph A ran: {sa}; frames {int(frames.sum())} ({frames.tolist()})", flush=True)
    t0 = time.time()
    na_consts = len(g.consts)
    a_top = -(-g.top // 4096) * 4096
    audio_t = kk.build_b(g, frames)
    pb = g.compile(g.ops[na:])
    print(f"graph B: {len(pb)} programs, compiled in {time.time() - t0:.1f} s, DRAM top "
          f"{g.top / 2**20:.0f} MiB", flush=True)
    # graph B's arena zero (its tensors' pad rows are read as zeros), then its constants
    run.write_at(a_top, np.zeros(-(-g.top // 4096) * 4096 - a_top, np.uint8))
    for at, arr in g.consts[na_consts:]:
        run.write_at(at, arr)
    run.write_at(kk.seed.addr + 4 * kk.seed.pad, np.array([a.seed], np.uint32))
    sb = run.run(pb, g.top)
    audio = run.read(g, audio_t).reshape(-1)
    wall = time.time() - t_all
    write_wav(a.out, audio)
    secs = len(audio) / 24000
    print(f"graph B ran: {sb}", flush=True)
    check(bool(np.isfinite(audio).all()) and float(np.abs(audio).max()) > 1e-3,
          f"{secs:.2f} s of audio, finite and not silent", f"peak {float(np.abs(audio).max()):.3f}; "
          f"{a.out}; wall {wall:.1f} s with the host's writes, real-time factor "
          f"{wall / secs:.2f}")
    if a.ref:
        import torch
        from kokoro.ref import KRef
        m = KRef.load(Path(a.weights) / "kokoro-v1_0.safetensors", a.config)
        noise = run.read(g, kk.t["noise"])
        out = m.forward_parts(ids, torch.as_tensor(style),
                              noise={"sine": noise, "rand_ini": np.zeros(9, np.float32)})
        check(out["pred_dur"].tolist() == frames.tolist(), "durations are the reference's",
              f"{frames.tolist()}")
        for k, lim in (("F0", 0.999), ("N", 0.999)):
            c = cos(run.read(g, kk.t[k]), out[k].numpy())
            check(c > lim, f"{k} curve cosine to the reference > {lim}", f"{c:.5f}")
        b = out["audio"].numpy()
        L = min(len(b), len(audio))
        A, B = logmel(audio[:L]), logmel(b[:L])
        c = cos(A - A.mean(), B - B.mean())
        check(c > 0.98, "log-mel spectrogram cosine to the reference > 0.98",
              f"{c:.4f}; mean |d log mel| {np.abs(A - B).mean():.3f}; waveform cosine "
              f"{cos(audio[:L], b[:L]):.3f} (phase: F0 a few Hz apart)")
        write_wav(Path(a.out).with_suffix(".ref.wav"), b)
    if ecc0 is not None:
        from opentpu.host import memcal
        e1 = memcal.ecc(board.t)
        check(e1 == ecc0, "ECC counters unchanged (else: not trusted)", f"{ecc0} -> {e1}")
    print(f"RESULT: {'PASS' if not fails else 'FAIL'} ({fails} failed)", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
