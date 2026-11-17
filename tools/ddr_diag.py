"""DDR3 diagnostics on the card for ECC errors that show under some workloads only (r5-m8,
2026-10-26: channel 0 under BitNet's batched decode steps, never under one sequence's decode).

    python tools/ddr_diag.py pattern [--mib 512] [--load none|fp4|ternary:4] [--passes 2]
    python tools/ddr_diag.py steps MODEL_DIR [--n 32]
    python tools/ddr_diag.py rmw [--iters 200000]
    python tools/ddr_diag.py wr [--mib 256] [--load ternary:4]
    python tools/ddr_diag.py lanes [--load ternary:4] [--reps 20]

lanes: which lanes and bits read wrong under the core's load at the taps the PHY has now (the
calibration's, or loadcal's): LiteDRAM's BIST writes its data to the last quarter of each channel,
then reads it back reps times, idle and while the core runs the load; per channel and lane the
beats read wrong, the data bits of the lane seen wrong (OR over the reads), and how many reads
had any. The core's own reads do not count here (the ECC counters do; printed too). The BIST
writes without ECC check bits: zeros over DRAM after it (Board.scrub).

wr: is a wrong word written wrong or read wrong, and does it need the core's load? Per channel
random data is written by the host and read back twice, in four phases: written idle or under
the core's load, read idle or under it. A word wrong in both reads was stored wrong (the write,
or the cell); a word wrong in one read only was read wrong. Per phase and channel: the words
wrong in the first / second / both reads, the byte lanes and data bits they differ in, the
first addresses, the ECC counters. The host sees data after the ECC, so a word read wrong here
is an uncorrectable (or miscorrected) one; the corrected ones are in the counters only. The
counters of a phase under load include the core's own reads: the first line is the load alone.

rmw: the accelerator streams 4 MiB of DRAM (64 KiB tiles, every DRAM row in turn) and between
two tile loads stores to one 64-byte beat: nothing (reads only), a whole beat (64 B), one 4-byte
word (a partial beat: the channel module's read-modify-write), or two 4-byte words of the same
beat back to back (as the batched steps' picks were); each at a beat on channel 0 and one on
channel 1 (Board.beat_at); or the loaded 64 KiB tile stored back whole (long bursts of whole
beats on both channels, a read / write turnaround each). Reports the ECC counters per case: which traffic reads back wrong.

pattern: per channel, random data written by the host (PCIe DMA) at channel offsets from 1 GiB,
read back and compared; idle (--load none) or while the core runs loadcal's load (its MXU work
and DMA; OTPU_LOAD_KIND's kinds), whose image sits at the bottom of DRAM, far below. Reports per
channel the 64-bit words read wrong, the byte lanes they differ in, and the ECC counters
(corrected / uncorrectable) after. OVERWRITES the card's DRAM there (nothing of a model lives on
the card between runs; Board.scrub after it).

steps: a BitNet (or any model the vLLM worker builds) engine with 4 KV slots and the batch area;
the decode in four ways, the ECC counters read and cleared after every run: one sequence's
resident decode step (step), 4 rows as one rows program with logits to the host (run_rows),
4 and 2 rows as llm/batch.py's steps (picked on the card). Reports per mode the runs, the runs
with ECC errors, corrected / uncorrectable. Tokens are not checked (the counters are the point).
Then (on the card) run_rows split in phases -- the run alone (the host's logits read stubbed out)
and three host reads of its logits -- with a verdict: the core's reads, the host's read path or
stored wrong (split_verdict).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _ecc(t, clear=True):
    from opentpu.host import memcal
    if t is None:
        return [(0, 0), (0, 0)]
    return memcal.ecc(t, clear=clear)


def cmd_pattern(a) -> int:
    from opentpu.host import loadcal
    from opentpu.host.board import Board, XdmaTransport, device_config
    b = Board(XdmaTransport(a.dev), check=False)
    t, info = b.t, b.info()
    print(f"card build {info.get('build_id') or 0:#x}, core {info.get('core_khz')} kHz, "
          f"temp {info.get('temp_c')}; load {a.load}; {a.mib} MiB a channel, {a.passes} passes")
    b.scrub(force=True)
    running = False
    if a.load != "none":
        kind, rows = loadcal.load_kind(a.load)
        cfg = device_config(info, DRAM_BYTES=1 << 30)
        img, lay, prog = loadcal.build(cfg, rows=rows, kind=kind)
        b.write(0, img)
        b.load(lay["prog"], prog)
        _ecc(t)
        b.start()
        running = True
    else:
        _ecc(t)
    rng = np.random.default_rng(a.seed)
    piece = 64 << 20
    bad = {0: 0, 1: 0}
    lanes = {0: np.zeros(8, np.int64), 1: np.zeros(8, np.int64)}
    t0 = time.time()
    try:
        for p in range(a.passes):
            for ch in (0, 1):
                for off in range(1 << 30, (1 << 30) + (a.mib << 20), piece):
                    d = rng.integers(0, 256, piece, np.uint8)
                    t.mem_write(ch, off, d)
                    r = t.mem_read(ch, off, piece)
                    w = (d.view(np.uint64) != r.view(np.uint64))
                    if w.any():
                        bad[ch] += int(w.sum())
                        x = (d.reshape(-1, 8) != r.reshape(-1, 8))
                        lanes[ch] += x.sum(0)
    finally:
        if running:
            b.stop()
    e = _ecc(t)
    dt = time.time() - t0
    gb = a.passes * 2 * 2 * (a.mib << 20) / 1e9
    for ch in (0, 1):
        print(f"channel {ch}: {bad[ch]} words read wrong; byte lanes {lanes[ch].tolist()}; "
              f"ECC corrected / uncorrectable {e[ch][0]} / {e[ch][1]}")
    print(f"{gb:.1f} GB moved in {dt:.0f} s; temp {b.info().get('temp_c')}")
    b.scrub(force=True)
    return 0 if not any(bad.values()) and not any(x[1] for x in e) else 1


def _piece(seed, ch, off, n):
    return np.random.default_rng([seed, ch, off]).integers(0, 256, n, np.uint8)


def wr_write(t, ch, offs, piece, seed):
    for off in offs:
        t.mem_write(ch, off, _piece(seed, ch, off, piece))


def wr_read(t, ch, offs, piece, seed):
    """Two reads of what wr_write wrote: words wrong in the first, the second, both; the byte
    lanes and the bits (of the 64-bit word) the first read differs in; the first addresses."""
    r = {"first": 0, "second": 0, "both": 0, "lanes": np.zeros(8, np.int64),
         "bits": np.zeros(64, np.int64), "at": []}
    for off in offs:
        d = _piece(seed, ch, off, piece).view(np.uint64)
        x1 = d ^ np.asarray(t.mem_read(ch, off, piece)).view(np.uint64)
        x2 = d ^ np.asarray(t.mem_read(ch, off, piece)).view(np.uint64)
        w1, w2 = x1 != 0, x2 != 0
        r["first"] += int(w1.sum())
        r["second"] += int(w2.sum())
        r["both"] += int((w1 & w2).sum())
        if w1.any():
            x = x1[w1]
            r["lanes"] += (x.view(np.uint8).reshape(-1, 8) != 0).sum(0, dtype=np.int64)
            r["bits"] += np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1, bitorder="little").sum(0, dtype=np.int64)
            r["at"] += [(off + 8 * int(i), int(v)) for i, v in zip(np.flatnonzero(w1)[:4], x[:4])]
    r["at"] = r["at"][:4]
    return r


def wr_verdict(r):
    if not r["first"] and not r["second"]:
        return "clean"
    if r["both"] == r["first"] == r["second"]:
        return "stored wrong (the write, or the cell)"
    if not r["both"]:
        return "read wrong (the stored word is right)"
    return "both: stored wrong and read wrong"


def cmd_wr(a) -> int:
    from opentpu.host import loadcal
    from opentpu.host.board import Board, XdmaTransport, device_config
    b = Board(XdmaTransport(a.dev), check=False)
    t, info = b.t, b.info()
    print(f"card build {info.get('build_id') or 0:#x}, core {info.get('core_khz')} kHz, "
          f"temp {info.get('temp_c')}; load {a.load}; {a.mib} MiB a channel")
    b.scrub(force=True)
    kind, rows = loadcal.load_kind(a.load)
    img, lay, prog = loadcal.build(device_config(info, DRAM_BYTES=1 << 30), rows=rows, kind=kind)
    b.write(0, img)
    state = {"on": False}

    def load(on):
        if on and not state["on"]:
            b.load(lay["prog"], prog)
            b.start()
        elif state["on"] and not on:
            b.stop()
        state["on"] = on

    piece = 64 << 20
    offs = range(1 << 30, (1 << 30) + (a.mib << 20), piece)
    bad = 0
    try:
        _ecc(t)
        load(True)
        time.sleep(a.alone)
        load(False)
        e = _ecc(t)
        print(f"the load alone, {a.alone} s: ECC ch0 {e[0][0]} / {e[0][1]}, ch1 {e[1][0]} / {e[1][1]}")
        for wl in (False, True):
            for rl in (False, True):
                for ch in (0, 1):
                    load(wl)
                    wr_write(t, ch, offs, piece, a.seed)
                    load(False)
                    _ecc(t)
                    load(rl)
                    r = wr_read(t, ch, offs, piece, a.seed)
                    load(False)
                    e = _ecc(t)
                    bad += r["first"] + r["second"] + sum(x[1] for x in e)
                    print(f"write {'load' if wl else 'idle'} / read {'load' if rl else 'idle'}, channel {ch}: "
                          f"wrong in read 1 / read 2 / both {r['first']} / {r['second']} / {r['both']}: "
                          f"{wr_verdict(r)}; ECC during the reads ch0 {e[0][0]} / {e[0][1]}, ch1 {e[1][0]} / {e[1][1]}")
                    if r["first"]:
                        print(f"    byte lanes {r['lanes'].tolist()}; bits {np.flatnonzero(r['bits']).tolist()[:24]}; "
                              f"first {[(hex(o), hex(v)) for o, v in r['at']]}")
    finally:
        load(False)
    print(f"temp {b.info().get('temp_c')}")
    b.scrub(force=True)
    return 0 if not bad else 1


def lane_bits(bits: int, nm: int = 9) -> dict:
    """bist_lane_bits (bit 8m + b: data bit b of lane m read wrong) -> {lane: [bits]}."""
    return {m: [b for b in range(8) if bits >> (8 * m + b) & 1] for m in range(nm) if bits >> (8 * m) & 0xFF}


def lanes_line(ch, r, reps):
    bad = {m: n for m, n in enumerate(r["lanes"]) if n}
    return (f"channel {ch}: {sum(r['lanes'])} lane-beats wrong in {r['runs']} of {reps} reads; "
            f"by lane {bad or '-'}; bits {lane_bits(r['bits']) or '-'}")


def cmd_lanes(a) -> int:
    from opentpu.host import ddrcal, loadcal, memcal
    from opentpu.host.board import CH_BYTES, Board, XdmaTransport, device_config
    b = Board(XdmaTransport(a.dev), check=False)
    t, info = b.t, b.info()
    c = memcal.csr(t, memcal.DATA)
    chans = [ddrcal.Chan(c, ch) for ch in (0, 1)]
    hz = loadcal.sys_hz(memcal.DATA)
    ch_beats = ((info.get("cfg") or {}).get("channel_bytes") or CH_BYTES) // 64
    base = ch_beats * 3 // 4
    beats = min(ch_beats // 4, a.mib << 14)
    print(f"card build {info.get('build_id') or 0:#x}, temp {info.get('temp_c')}; load {a.load}; "
          f"BIST {beats * 64 >> 20} MiB a channel x {a.reps} reads at the current taps")
    b.scrub(force=True)
    loadcal.bist_write(chans, base, beats, hz)
    kind, rows = loadcal.load_kind(a.load)
    img, lay, prog = loadcal.build(device_config(info, DRAM_BYTES=1 << 30), rows=rows, kind=kind)
    b.write(0, img)
    bad = 0
    for on in (False, True):
        acc = [{"lanes": [0] * 9, "bits": 0, "runs": 0} for _ in chans]
        _ecc(t)
        if on:
            b.load(lay["prog"], prog)
            b.start()
        try:
            for _ in range(a.reps):
                for k in chans:
                    loadcal._bist(k, base, beats, 1)
                for i, k in enumerate(chans):
                    r = ddrcal.bist_wait(k, beats, 1, hz, timeout=120)
                    acc[i]["lanes"] = [x + y for x, y in zip(acc[i]["lanes"], r["lanes"])]
                    acc[i]["bits"] |= r["bits"]
                    acc[i]["runs"] += bool(r["errors"])
        finally:
            if on and b.stop():
                print("   (the load ended before the reads did)")
        e = _ecc(t)
        print(f"{'under the load' if on else 'idle'}:")
        for i in range(len(chans)):
            print("   " + lanes_line(i, acc[i], a.reps))
            bad += sum(acc[i]["lanes"])
        print(f"   ECC counters meanwhile (the core's reads and the host's): ch0 {e[0][0]} / {e[0][1]}, "
              f"ch1 {e[1][0]} / {e[1][1]}")
    print(f"temp {b.info().get('temp_c')}")
    b.scrub(force=True)
    return 0 if not bad else 1


def cmd_steps(a) -> int:
    from opentpu.llm import batch as BA
    from opentpu.host import chat as CH
    from opentpu.llm import load_spec, qwen3
    from opentpu.llm.qwen3 import Engine
    spec = load_spec(a.model)
    if getattr(spec, "subln", False):              # BitNet: the ternary TMM image
        from opentpu.llm.bitnet import IMAGE, Weights
        W, kw = Weights(a.model), dict(IMAGE)
    else:
        W, kw = qwen3.load_weights(a.model), dict(wformat="int8", head_format=None)
    be, cfg = CH.make_backend(a.backend, spec, 512, a.dev, Path(a.model).name, lookup=True,
                              wformat=kw["wformat"], head_format=kw.get("head_format"))
    if cfg is None:                                 # the ISA simulator
        from opentpu.isasim import board_config
        cfg = board_config(DRAM_BYTES=1 << 28)
    eng = Engine(spec, W, cap=512, cfg=cfg, backend=be, resident=True, batch=4,
                 batch_area=True, **kw)
    board = getattr(eng.backend, "board", None)     # (None: the ISA simulator, a dry run)
    t = board.t if board is not None else None
    temp = (lambda: board.info().get("temp_c")) if board is not None else (lambda: None)
    st = BA.Steps(eng)
    print(f"{Path(a.model).name}: 4 KV slots, batch area; temp {temp()}")
    prompts = [[1, 791, 6342, 315], [1, 40, 1093, 311, 3371], [1, 3923, 374], [1, 2675, 527, 264]]
    for s, p in enumerate(prompts):
        try:
            for _ in eng.prefill_chunks(p, seq=s, logits=False):
                pass
        except RuntimeError as e:
            if "ECC" not in str(e):
                raise
    _ecc(t)
    res = {}

    def one(mode, fn):
        r = res.setdefault(mode, [0, 0, 0, 0])      # runs, runs with errors, corrected, ded
        try:
            fn()
        except RuntimeError as e:
            if "ECC" not in str(e):
                raise
        c = _ecc(t)
        r[0] += 1
        n0 = c[0][0] + c[1][0]
        n1 = c[0][1] + c[1][1]
        r[1] += bool(n0 or n1)
        r[2] += n0
        r[3] += n1
        if n0 or n1:
            r.append((c[0], c[1]))

    tok = 13
    for _ in range(a.n):
        one("step (one sequence, resident decode)", lambda: eng.step(tok))
    for _ in range(a.n // 2):
        def rows():
            lg = eng.run_rows([(s, eng.poss[s]) for s in range(4)], [tok] * 4, list(range(4)))
            for s in range(4):
                eng.poss[s] += 1
            return lg
        one("run_rows (4 rows, logits to the host)", rows)
    # run_rows split in phases: the run with the host's logits read stubbed out, then the
    # host reads the logits three times, the counters read and cleared after each phase. Errors
    # in the run: the core's reads; only in the first read: the host's read path (C2H);
    # in every read: the logits were stored wrong (the write path)
    split = {k: [0, 0, 0] for k in ("run", "read 1", "read 2", "read 3")}
    io, v, vl, S = eng.image.io, spec.vocab, eng.image.v_loc, eng.cfg.S

    def logits_read():
        for r in range(4):
            for s in range(S):
                eng.backend.read(s, io["logits"] + 4 * (r * v + s * vl), 4 * vl)

    def phase(k, fn):
        try:
            fn()
        except RuntimeError as e:
            if "ECC" not in str(e):
                raise
        c = _ecc(t)
        n0, n1 = c[0][0] + c[1][0], c[0][1] + c[1][1]
        x = split[k]
        x[0] += bool(n0 or n1)
        x[1] += n0
        x[2] += n1
        if (n0 or n1) and len(x) < 7:
            x.append((c[0], c[1]))

    def rows_run_only():
        rows = [(s, eng.poss[s]) for s in range(4)]
        real = eng.backend.read
        eng.backend.read = lambda s, a, n: np.zeros(n, np.uint8)
        try:
            eng.run_rows(rows, [tok] * 4, list(range(4)))
        finally:
            eng.backend.read = real
            for s in range(4):
                eng.poss[s] += 1

    if board is not None:
        for _ in range(a.n):
            phase("run", rows_run_only)
            for k in ("read 1", "read 2", "read 3"):
                phase(k, logits_read)

    def steps(n):
        rows = [(s, eng.poss[s]) for s in range(n)]
        try:
            return st.run(rows, [tok] * n)
        except RuntimeError:            # the run went (an ECC error after it): its rows moved on
            for s, p in rows:
                eng.poss[s] = p + 1
            raise
    for _ in range(a.n):
        one("steps (4 rows, picked on the card)", lambda: steps(4))
    for _ in range(a.n):
        one("steps (2 rows)", lambda: steps(2))
    for mode, r in res.items():
        ch = [x for x in r[4:]][:4]
        print(f"{mode}: {r[0]} runs, {r[1]} with ECC errors, corrected {r[2]}, uncorrectable "
              f"{r[3]}" + (f"; first per channel {ch}" if ch else ""))
    if board is not None:
        print("run_rows split (" + str(a.n) + " runs; runs with errors, corrected, uncorrectable"
              "): " + "; ".join(f"{k} {x[0]} / {x[1]} / {x[2]}" + (f" {x[3:]}" if x[3:] else "")
                                for k, x in split.items()))
        print("run_rows split verdict: " + split_verdict(split))
    print(f"temp {temp()}")
    close = getattr(eng.backend, "close", None)
    if close is not None:
        close()
    return 0


def split_verdict(split) -> str:
    """run_rows split's phases -> where the errors come from."""
    run, r1 = split["run"][0], split["read 1"][0]
    again = split["read 2"][0] + split["read 3"][0]
    if not (run or r1 or again):
        return "clean"
    out = []
    if run:
        out.append("the core's reads during the run")
    if again:
        out.append("stored wrong (re-reads fail too: the write path)")
    elif r1:
        out.append("the host's first read only (the C2H read path, transient)")
    return "; ".join(out)


def _rmw_kernel():
    from opentpu import language as ol

    @ol.jit
    def k(m, n_iter: int, kind: str):
        t = ol.empty([16384])
        s = ol.load(m.src)
        for _ in ol.range(n_iter):
            for j in range(4):
                ol.load(m.big[j, :], out=t)
                if kind == "full":
                    ol.store(m.dst, s)
                elif kind == "partial":
                    ol.store(m.dst[0:1], s[0:1])
                elif kind == "two":
                    ol.store(m.dst[0:1], s[0:1])
                    ol.store(m.dst[1:2], s[1:2])
                elif kind == "tile":           # 64 KiB of whole beats, both channels
                    ol.store(m.out[j, :], t)
    return k


TROWS = 64


def cmd_rmw(a) -> int:
    from opentpu.host.board import Board, XdmaTransport, beat_at, device_config
    from opentpu.llm.qwen3 import _tdesc
    b = Board(XdmaTransport(a.dev), check=False)
    t, info = b.t, b.info()
    cfg = device_config(info, DRAM_BYTES=1 << 30)
    chash = bool((info.get("caps") or {}).get("chash"))
    big, src = 1 << 22, 1 << 20                 # 4 MiB streamed at 4 MiB; the source words at 1 MiB
    dsts = {}
    for addr in range(16 << 20, (16 << 20) + 512, 64):    # one beat on each channel
        dsts.setdefault(beat_at(addr, chash)[0], addr)
    prog_at = 32 << 20
    rng = np.random.default_rng(1)
    img = np.zeros(prog_at, np.uint8)
    img[big:big + (4 << 20)] = rng.integers(0, 256, 4 << 20, np.uint8)
    img[src:src + 64] = rng.integers(0, 256, 64, np.uint8)
    b.scrub(force=True)
    b.write(0, img)
    print(f"card build {info.get('build_id') or 0:#x}, temp {info.get('temp_c')}; chash {chash}; "
          f"{a.iters} iterations of 4 x 64 KiB loads; beats {dsts}")
    bad = 0
    for kind in ("none", "full", "partial", "two", "tile"):
        for ch in ((0,) if kind in ("none", "tile") else (0, 1)):
            m = SimpleNamespace(big=_tdesc(big, (TROWS, 16384)), src=_tdesc(src, (16,)),
                                dst=_tdesc(dsts[ch], (16,)), out=_tdesc(24 << 20, (4, 16384)))
            prog = _rmw_kernel().trace(cfg, 0, {"m": m, "n_iter": a.iters, "kind": kind}).finish()
            b.load(prog_at, prog)
            _ecc(t)
            t0 = time.time()
            try:
                b.start()
                st = b.wait(timeout=600)
                err = ""
            except RuntimeError as e:
                st, err = {}, " (" + str(e)[:60] + ")"
            e = _ecc(t)
            bad += sum(x[0] + x[1] for x in e)
            print(f"{kind:8s} store beat on channel {ch if kind not in ('none', 'tile') else '-'}: "
                  f"ECC ch0 {e[0][0]} / {e[0][1]}, ch1 {e[1][0]} / {e[1][1]}; "
                  f"{time.time() - t0:.1f} s{err}")
    print(f"temp {b.info().get('temp_c')}")
    b.scrub(force=True)
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("pattern")
    p.add_argument("--mib", type=int, default=512)
    p.add_argument("--load", default="none")
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dev", default="/dev/xdma0")
    s = sp.add_parser("steps")
    s.add_argument("model")
    s.add_argument("--n", type=int, default=32)
    s.add_argument("--backend", default="board", choices=["board", "isa"])
    s.add_argument("--dev", default="/dev/xdma0")
    r = sp.add_parser("rmw")
    r.add_argument("--iters", type=int, default=200000)
    r.add_argument("--dev", default="/dev/xdma0")
    w = sp.add_parser("wr")
    w.add_argument("--mib", type=int, default=256)
    w.add_argument("--load", default="ternary:4")
    w.add_argument("--alone", type=int, default=30)
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--dev", default="/dev/xdma0")
    n = sp.add_parser("lanes")
    n.add_argument("--load", default="ternary:4")
    n.add_argument("--reps", type=int, default=20)
    n.add_argument("--mib", type=int, default=128)
    n.add_argument("--dev", default="/dev/xdma0")
    a = ap.parse_args()
    return {"pattern": cmd_pattern, "steps": cmd_steps, "rmw": cmd_rmw, "wr": cmd_wr,
            "lanes": cmd_lanes}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
