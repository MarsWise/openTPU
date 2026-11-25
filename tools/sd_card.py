"""Stable Diffusion 1.5 text to image with every model operation on the device
(opentpu/sd_graph.py): CLIP (prompt and empty prompt), the initial latent from the seed (VOP
RAND), DDIM steps with classifier-free guidance (two UNet evaluations a step), the VAE decoder;
or with --lcm-lora (LCM-LoRA merged into the UNet's weights as they are read) LCMScheduler steps
(default 4, guidance 1: one UNet evaluation a step, each step's noise drawn on the device) and
with --taesd AutoencoderTiny's decoder instead of the VAE's.

    python tools/sd_card.py --sd DIR [--unet-shards 'DIR/unet_*.safetensors'] [--isa]
        [--prompt "..."] [--size 256] [--steps 20] [--seed 42] [--out sd.png] [--ref]
        [--lcm-lora pytorch_lora_weights.safetensors] [--taesd diffusion_pytorch_model.safetensors]

DIR: the diffusers folder (text_encoder/, unet/, vae/, tokenizer/; fp16 safetensors). The host:
the prompt's tokens (CLIP BPE), the schedule's constants, the seed; it writes the image (all
weights and a zero arena) once, starts one chained run, reads the image. --ref: the same
pipeline in diffusers (fp32, host CPU, a check: not part of the run) from the same initial
latent: the image's cosine and PSNR. Nothing is loaded into the FPGA. Prints CHECK lines."""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def reference(sd_dir, unet_files, ids_c, ids_u, z0, h, w, steps, gs=7.5):
    import torch
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from safetensors.torch import load_file
    from transformers import CLIPTextConfig, CLIPTextModel
    U = Path(sd_dir)
    c = CLIPTextConfig(**{k: v for k, v in json.loads((U / "text_encoder/config.json").read_text())
                          .items() if not k.startswith("_")})
    te = CLIPTextModel(c).eval()
    te.load_state_dict({k.replace("text_model.", ""): v.float() for k, v in
                        load_file(str(U / "text_encoder/model.fp16.safetensors")).items()},
                       strict=False)
    with torch.no_grad():
        ec = te(torch.tensor([ids_c])).last_hidden_state
        eu = te(torch.tensor([ids_u])).last_hidden_state
    del te
    m = UNet2DConditionModel.from_config(json.loads((U / "unet/config.json").read_text())).eval()
    from safetensors import safe_open
    for p in unet_files:                        # (a few hundred tensors at a time: fp32 on a
        with safe_open(str(p), "pt") as f:      # host of 8 GB)
            keys = list(f.keys())
            for i in range(0, len(keys), 64):
                m.load_state_dict({k: f.get_tensor(k).float() for k in keys[i:i + 64]},
                                  strict=False)
    sc = DDIMScheduler.from_config(json.loads((U / "scheduler/scheduler_config.json").read_text()))
    sc.set_timesteps(steps)
    x = torch.as_tensor(z0).reshape(1, h, w, 4).permute(0, 3, 1, 2)
    with torch.no_grad():
        for t in sc.timesteps:
            e_c = m(x, t, encoder_hidden_states=ec).sample
            e_u = m(x, t, encoder_hidden_states=eu).sample
            x = sc.step(e_u + gs * (e_c - e_u), t, x).prev_sample
    del m
    v = AutoencoderKL.from_config(json.loads((U / "vae/config.json").read_text())).eval()
    v.load_state_dict({k: t.float() for k, t in
                       load_file(str(U / "vae/diffusion_pytorch_model.fp16.safetensors")).items()})
    with torch.no_grad():
        return v.decode(x / 0.18215).sample[0].permute(1, 2, 0).numpy()


def reference_lcm(sd_dir, unet_files, lora, taesd, ids_c, ids_u, z0, h, w, steps, gs, seed):
    """diffusers' LCM pipeline steps (fp32, host CPU): the UNet with the LoRA merged (W + alpha
    / rank up @ down, sd_graph.lora_delta: diffusers needs peft to load it itself), LCMScheduler
    from SD 1.5's scheduler config, each step's noise the device's (rand: seed + 2 + step) in
    place of the scheduler's own draw, then TAESD's decoder (or the VAE's)."""
    import torch
    from diffusers import AutoencoderKL, AutoencoderTiny, LCMScheduler, UNet2DConditionModel
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import CLIPTextConfig, CLIPTextModel
    from opentpu import sd_graph as S
    from opentpu.kernels import randn as RN
    U = Path(sd_dir)
    c = CLIPTextConfig(**{k: v for k, v in json.loads((U / "text_encoder/config.json").read_text())
                          .items() if not k.startswith("_")})
    te = CLIPTextModel(c).eval()
    te.load_state_dict({k.replace("text_model.", ""): v.float() for k, v in
                        load_file(str(U / "text_encoder/model.fp16.safetensors")).items()},
                       strict=False)
    guided = gs > 1
    with torch.no_grad():
        ec = te(torch.tensor([ids_c])).last_hidden_state
        eu = te(torch.tensor([ids_u])).last_hidden_state if guided else None
    del te
    m = UNet2DConditionModel.from_config(json.loads((U / "unet/config.json").read_text())).eval()
    for p in unet_files:
        with safe_open(str(p), "pt") as f:
            keys = list(f.keys())
            for i in range(0, len(keys), 64):
                m.load_state_dict({k: f.get_tensor(k).float() for k in keys[i:i + 64]},
                                  strict=False)
    lf = safe_open(str(lora), "np")
    sdict = m.state_dict()
    with torch.no_grad():
        for k, mod in S.lora_targets(list(sdict), list(lf.keys())).items():
            al = lf.get_tensor(mod + ".alpha") if mod + ".alpha" in lf.keys() else None
            d = S.lora_delta(lf.get_tensor(mod + ".lora_down.weight"),
                             lf.get_tensor(mod + ".lora_up.weight"), al, tuple(sdict[k].shape))
            sdict[k].add_(torch.as_tensor(d))
    sc = LCMScheduler.from_config(json.loads((U / "scheduler/scheduler_config.json").read_text()))
    sc.set_timesteps(steps)
    ts = sc.timesteps.tolist()
    x = torch.as_tensor(z0).reshape(1, h, w, 4).permute(0, 3, 1, 2)
    with torch.no_grad():
        for i, t in enumerate(ts):
            e = m(x, t, encoder_hidden_states=ec).sample
            if guided:
                e_u = m(x, t, encoder_hidden_states=eu).sample
                e = e_u + gs * (e - e_u)
            den = sc.step(e, t, x).denoised
            if i < len(ts) - 1:
                ap = sc.alphas_cumprod[ts[i + 1]]
                z = torch.as_tensor(RN.seeded_reference(1, h * w * 4, seed, 2 + i)
                                    .reshape(1, h, w, 4)).permute(0, 3, 1, 2)
                x = ap.sqrt() * den + (1 - ap).sqrt() * z
            else:
                x = den
    del m
    with torch.no_grad():
        if taesd:
            v = AutoencoderTiny().eval()
            v.load_state_dict(load_file(str(taesd)))
            return v.decode(x).sample[0].permute(1, 2, 0).numpy()
        v = AutoencoderKL.from_config(json.loads((U / "vae/config.json").read_text())).eval()
        v.load_state_dict({k: t.float() for k, t in
                           load_file(str(U / "vae/diffusion_pytorch_model.fp16.safetensors")).items()})
        return v.decode(x / 0.18215).sample[0].permute(1, 2, 0).numpy()


def to_png(path, a):
    from PIL import Image
    Image.fromarray(((np.clip(a, -1, 1) + 1) * 127.5).round().astype(np.uint8)).save(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sd", required=True)
    ap.add_argument("--unet-shards", help="glob of UNet safetensors (default: DIR/unet's file)")
    ap.add_argument("--prompt", default="a photo of a red fox in the snow")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, help="default 20 (DDIM), 4 (LCM)")
    ap.add_argument("--guidance", type=float, help="default 7.5 (DDIM), 1 (LCM: no guidance)")
    ap.add_argument("--lcm-lora", help="LCM-LoRA (kohya safetensors): LCMScheduler steps")
    ap.add_argument("--taesd", help="AutoencoderTiny's safetensors: its decoder for the VAE's")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="sd.png")
    ap.add_argument("--isa", action="store_true")
    ap.add_argument("--profile", action="store_true",
                    help="on the card: each program run on its own, cycles by op family")
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--ref", action="store_true")
    ap.add_argument("--no-cache", action="store_true",
                    help="build and compile anew, keep nothing (else: resident.PlanCache)")
    a = ap.parse_args()
    sampler = "lcm" if a.lcm_lora else "ddim"
    decoder = "taesd" if a.taesd else "vae"
    if a.steps is None:
        a.steps = 4 if a.lcm_lora else 20
    if a.guidance is None:
        a.guidance = 1.0 if a.lcm_lora else 7.5
    evals = a.steps * (2 if sampler == "ddim" or a.guidance > 1 else 1)
    from transformers import CLIPTokenizer
    from opentpu import resident as RG
    from opentpu import sd_graph as S
    from opentpu.kernels import randn as RN
    fails = 0

    def check(ok, what, detail=""):
        nonlocal fails
        print(f"CHECK {'PASS' if ok else 'FAIL'}: {what}" + (f" ({detail})" if detail else ""),
              flush=True)
        fails += 0 if ok else 1

    U = Path(a.sd)
    unet = sorted(glob.glob(a.unet_shards)) if a.unet_shards else \
        [str(U / "unet/diffusion_pytorch_model.fp16.safetensors")]
    W = S.Weights(text=U / "text_encoder/model.fp16.safetensors", unet=unet,
                  vae=U / "vae/diffusion_pytorch_model.fp16.safetensors", taesd=a.taesd,
                  lora=a.lcm_lora)
    print(f"sampler {sampler} ({a.steps} steps, guidance {a.guidance}: {evals} UNet "
          f"evaluations), decoder {decoder}" +
          (f", LoRA {Path(a.lcm_lora).name}: {len(W.lora)} UNet weights" if a.lcm_lora else ""),
          flush=True)
    tok = CLIPTokenizer(str(U / "tokenizer/vocab.json"), str(U / "tokenizer/merges.txt"))
    ids_c = tok(a.prompt, padding="max_length", max_length=77, truncation=True).input_ids
    ids_u = tok("", padding="max_length", max_length=77).input_ids
    h = w = a.size // 8
    ecc0 = None
    if a.isa:
        from opentpu.isasim import lean_config          # the card's configuration
        cfg = lean_config(DRAM_BYTES=1 << 32)
        board = None
    else:
        from opentpu.host import memcal
        from opentpu.host.board import Board, XdmaTransport, device_config
        board = Board(XdmaTransport(a.dev), check=False)
        info = board.info()
        cfg = device_config(info)
        for need in ("TFN", "VRAND"):
            if not getattr(cfg, need, False):
                raise SystemExit(f"the bitstream has no {need}")
        ecc0 = memcal.ecc(board.t)
        print(f"card: build {info.get('build_id') or 0:08x}, ECC {ecc0}", flush=True)
    cache = hit = key = None
    if board is not None and not a.profile and not a.no_cache:
        from opentpu.target import arch_hash
        cache = RG.PlanCache("sd15")
        key = cache.key("txt2img", 2, arch_hash(cfg), info.get("build_id"), cfg.DRAM_BYTES,
                        a.size, a.steps, a.guidance, sampler, decoder,
                        RG.files_id(U / "text_encoder/model.fp16.safetensors", *unet,
                                    U / "vae/diffusion_pytorch_model.fp16.safetensors",
                                    *[p for p in (a.lcm_lora, a.taesd) if p]),
                        RG.source_hash(__file__))
        t0 = time.time()
        hit = cache.load(key)
        if hit is not None:
            print(f"plan cache hit ({key}): image, programs and chain slots in "
                  f"{time.time() - t0:.0f} s, no build or compile", flush=True)
    t0 = time.time()
    if hit is None:
        g = S.SDGraph(cfg, 3 << 29)                     # constants from 1.5 GiB
        sd = S.SD(W)
        img_t, seed_t, _ = S.build_txt2img(g, sd, ids_c, ids_u, h, w, a.steps, a.guidance,
                                           sampler=sampler, decoder=decoder)
        print(f"{len(g.ops)} ops ({evals} UNet evaluations), built in {time.time() - t0:.0f} s; "
              f"activations {g.top / 2**20:.0f} MiB, weights {(g.ctop - g.cbase) / 2**20:.0f} MiB",
              flush=True)
        img = g.image()
        g.consts.clear()                                # (in the image now)
        ops, g.ops = g.ops, []
    else:
        img = RG.dense_image(hit["img"])
        img_t, seed_t = hit["out"], hit["seed"]
    RG.Graph.put(None, img, seed_t, np.array([a.seed], np.uint32).view(np.float32))
    t0 = time.time()
    if board is None:
        n, cache_isa = 0, {}
        for i in range(0, len(ops), 64):                # compiled and run a chunk at a time
            img, k = RG.run_isa(cfg, S.compile_cached(g, ops[i:i + 64], cache_isa), img)
            n += k
        st = {"instructions": n}
        out = g.read(img, S._dt(img_t))
    else:
        at = -(-(len(img) + (1 << 20)) // 4096) * 4096
        if a.profile:                                   # each program on its own: its cycles
            from opentpu import isa as I
            named, asm = [], {}
            for n_, p in S.compile_cached(g, ops, {}):
                if id(p) not in asm:
                    asm[id(p)] = np.asarray(I.assemble(p), np.uint32)
                named.append((n_, p))
            words = [asm[id(p)] for _, p in named]
            print(f"{len(words)} programs compiled in {time.time() - t0:.0f} s", flush=True)
            t0 = time.time()
            board.write(0, img)
            t_w = time.time() - t0
            rows = RG.profile_board(board, named, at, words=words)
            print(RG.profile_table(rows, top=30), flush=True)
            st = {"programs": len(rows), "cycles": sum(c for _, c in rows), "profiled": True}
        else:
            # the two prompts' token rows (clip.tok: the only programs a prompt changes) run on
            # their own before the chain; the rest is the plan, cached between invocations
            if hit is None:
                tok = [i for i, op in enumerate(ops) if op[0].startswith("clip.tok")]
                tok_args = [{k: v for k, v in ops[i][2].items() if k != "moves"} for i in tok]
                rest = [op for i, op in enumerate(ops) if i not in set(tok)]
                words = RG.compile_words(cfg, rest)
                print(f"{len(words) + len(tok)} programs compiled in {time.time() - t0:.0f} s",
                      flush=True)
                plan = RG.ChainPlan(board, None, at, words=words)
                if cache is not None:
                    t1 = time.time()
                    ok = cache.save(key, {"img": RG.sparse_image(img), "parts": plan.parts,
                                          "at": at, "stats": plan.stats, "tok": tok_args,
                                          "out": img_t, "seed": seed_t})
                    print(f"plan cache {'saved' if ok else 'NOT saved (disk)'} ({key}) in "
                          f"{time.time() - t1:.0f} s", flush=True)
            else:
                tok_args = hit["tok"]
                plan = RG.ChainPlan.from_parts(hit["parts"], hit["at"], hit["stats"])
            tg = RG.Graph(cfg)
            for args, ids in zip(tok_args, (ids_c, ids_u)):
                tg.op("clip.tok", RG.copies_k, moves=tuple((int(i), t, 0)
                                                           for t, i in enumerate(ids)), **args)
            tok_plan = RG.ChainPlan(board, None, at, words=RG.compile_words(cfg, tg.ops))
            t0 = time.time()
            board.write(0, img)
            t_w = time.time() - t0
            st0 = tok_plan.run(board, timeout=60.0)
            st = {**plan.stats, **plan.run(board, timeout=3600.0)}
            st["cycles"] += st0["cycles"]
            st["programs"] += len(tg.ops)
        st["write_s"] = round(t_w, 1)
        t_ = img_t
        raw = board.read(t_.addr, 4 * t_.H * t_.W * t_.C).view(np.float32)
        out = raw.reshape(t_.H * t_.W, t_.C)
    wall = time.time() - t0
    im = out.reshape(a.size, a.size, 3)
    to_png(a.out, im)
    check(bool(np.isfinite(im).all()) and float(im.std()) > 0.05,
          f"a {a.size} px image, finite and not flat", f"{a.out}; {st}; wall {wall:.0f} s")
    if a.ref:
        try:
            import diffusers  # noqa: F401
        except ImportError:
            print("REFERENCE NOT RUN: diffusers is not installed for this Python "
                  "(pip install diffusers==0.43.0); compare the PNG elsewhere", flush=True)
            a.ref = False
    if a.ref:
        z0 = RN.seeded_reference(1, h * w * 4, a.seed, 1).reshape(h * w, 4)
        if sampler == "lcm":
            ref = reference_lcm(U, unet, a.lcm_lora, a.taesd, ids_c, ids_u, z0, h, w, a.steps,
                                a.guidance, a.seed)
        elif decoder == "vae":
            ref = reference(U, unet, ids_c, ids_u, z0, h, w, a.steps, a.guidance)
        else:
            raise SystemExit("--ref with DDIM and --taesd: not implemented")
        c = float((im * ref).sum() / np.linalg.norm(im) / np.linalg.norm(ref))
        mse = float(((np.clip(im, -1, 1) - np.clip(ref, -1, 1)) ** 2).mean())
        psnr = 10 * np.log10(4 / max(mse, 1e-12))
        to_png(Path(a.out).with_suffix(".ref.png"), ref)
        check(c > 0.9, "the image's cosine to diffusers' (fp32, the same initial latent) > 0.9",
              f"{c:.4f}, PSNR {psnr:.1f} dB")
    if ecc0 is not None:
        from opentpu.host import memcal
        e1 = memcal.ecc(board.t)
        check(e1 == ecc0, "ECC counters unchanged (else: not trusted)", f"{ecc0} -> {e1}")
    print(f"RESULT: {'PASS' if not fails else 'FAIL'} ({fails} failed)", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
