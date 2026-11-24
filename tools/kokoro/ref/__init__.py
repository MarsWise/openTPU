"""The reference Kokoro-82M v1.0 graph: hexgrad's kokoro 0.9.4 (PyPI wheel, Apache-2.0; modules.py,
istftnet.py, custom_stft.py vendored unchanged but for the package-relative import and an unused
`attr` import), and `KRef`: KModel.forward_with_tokens (model.py) on a weights_only state dict,
with the SineGen / source noise taken from arguments, so the device graph can be compared with it
on the same noise. Used by tests and tools only (torch, transformers)."""
from __future__ import annotations

import json
import math

import torch
from transformers import AlbertConfig

from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder


class KRef(torch.nn.Module):
    def __init__(self, config: dict, state: dict):
        super().__init__()
        self.vocab = config["vocab"]
        self.bert = CustomAlbert(AlbertConfig(vocab_size=config["n_token"], **config["plbert"]))
        self.bert_encoder = torch.nn.Linear(self.bert.config.hidden_size, config["hidden_dim"])
        self.predictor = ProsodyPredictor(style_dim=config["style_dim"], d_hid=config["hidden_dim"],
                                          nlayers=config["n_layer"], max_dur=config["max_dur"],
                                          dropout=config["dropout"])
        self.text_encoder = TextEncoder(channels=config["hidden_dim"],
                                        kernel_size=config["text_encoder_kernel_size"],
                                        depth=config["n_layer"], n_symbols=config["n_token"])
        self.decoder = Decoder(dim_in=config["hidden_dim"], style_dim=config["style_dim"],
                               dim_out=config["n_mels"], **config["istftnet"])
        for key, sd in state.items():           # (model.py's loading: "module." prefixes)
            m = getattr(self, key)
            try:
                m.load_state_dict(sd)
            except RuntimeError:
                m.load_state_dict({k[7:]: v for k, v in sd.items()}, strict=False)
        self.eval()

    @staticmethod
    def load(safetensors_path, config_json) -> "KRef":
        """tools/kokoro/convert.py's kokoro-v1_0.safetensors (<module>.<key>, fp32)."""
        from safetensors.torch import load_file
        cfg = json.loads(open(config_json, encoding="utf-8").read())
        state: dict = {}
        for k, v in load_file(str(safetensors_path)).items():
            mod, key = k.split(".", 1)
            state.setdefault(mod, {})[key] = v
        return KRef(cfg, state)

    def ids(self, phonemes: str) -> list:
        return [0] + [self.vocab[p] for p in phonemes if p in self.vocab] + [0]

    @torch.no_grad()
    def forward_parts(self, input_ids, ref_s, speed: float = 1.0, noise=None):
        """model.py forward_with_tokens, returning its intermediates. noise: dict with 'sine'
        [Lup, 9] (SineGen's randn) and 'rand_ini' [9]; the source branch's noise is unused by
        the generator (SourceModuleHnNSF's `noise` output is discarded)."""
        ids = torch.LongTensor([input_ids])
        ref_s = torch.as_tensor(ref_s, dtype=torch.float32).reshape(1, -1)
        T = ids.shape[1]
        L = torch.full((1,), T, dtype=torch.long)
        mask = torch.zeros(1, T, dtype=torch.bool)
        bert = self.bert(ids, attention_mask=(~mask).int())
        d_en = self.bert_encoder(bert).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, L, mask)
        x, _ = self.predictor.lstm(d)
        duration = torch.sigmoid(self.predictor.duration_proj(x)).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze(0)
        idx = torch.repeat_interleave(torch.arange(T), pred_dur)
        aln = torch.zeros((T, idx.shape[0]))
        aln[idx, torch.arange(idx.shape[0])] = 1
        aln = aln.unsqueeze(0)
        en = d.transpose(-1, -2) @ aln
        F0, N = self.predictor.F0Ntrain(en, s)
        t_en = self.text_encoder(ids, L, mask)
        asr = t_en @ aln
        gen = self.decoder.generator
        sg = gen.m_source.l_sin_gen
        saved = (torch.rand, torch.randn_like)
        if noise is not None:
            ini, sn = torch.as_tensor(noise["rand_ini"]), torch.as_tensor(noise["sine"])
            torch.rand = lambda *a, **k: ini.reshape(1, -1).clone()
            calls = []

            def randn_like(t, *a, **k):
                calls.append(t.shape)
                return sn.reshape(t.shape).clone() if len(calls) == 1 else torch.zeros_like(t)
            torch.randn_like = randn_like
        try:
            audio = self.decoder(asr, F0, N, ref_s[:, :128]).squeeze()
        finally:
            torch.rand, torch.randn_like = saved
        return dict(bert=bert[0], d_en=d_en[0].T, d=d[0], dur_x=x[0], duration=duration[0],
                    pred_dur=pred_dur, en=en[0].T, F0=F0[0], N=N[0], t_en=t_en[0].T,
                    asr=asr[0].T, audio=audio, sg=sg)


def voice_row(pack, n_phonemes: int):
    """KPipeline's style vector: the voice pack's row for the phoneme count, [256]."""
    return torch.as_tensor(pack)[n_phonemes - 1].reshape(-1)


__all__ = ["KRef", "voice_row", "math"]
