import numpy as np
import pytest

from opentpu import Config, isa as I
from opentpu.bitnet import (fake_quant8, hf_to_tmm, pack_hf_bitnet, reference_linear,
                            unpack_hf_bitnet)
from opentpu.isasim import Machine
from opentpu.quant import pack_ternary_mxu


def test_hf_pack_matches_transformers_layout():
    w = np.array([[-1, 0], [0, 1], [1, -1], [1, 1]], np.int8)      # N=4, K=2
    packed = pack_hf_bitnet(w)
    assert packed.shape == (1, 2)
    # byte k = codes of rows 0..3, low pair first, code = value + 1
    assert packed[0, 0] == 0 | (1 << 2) | (2 << 4) | (2 << 6)
    assert packed[0, 1] == 1 | (2 << 2) | (0 << 4) | (2 << 6)


def test_hf_round_trip_and_reserved_code():
    rng = np.random.default_rng(1)
    w = rng.choice(np.array([-1, 0, 1], np.int8), (64, 40))
    assert np.array_equal(unpack_hf_bitnet(pack_hf_bitnet(w)), w)
    bad = pack_hf_bitnet(w)
    bad[0, 0] |= 3
    with pytest.raises(ValueError, match="code 3"):
        unpack_hf_bitnet(bad)


def test_hf_to_tmm_is_the_tmm_packing_of_the_same_values():
    rng = np.random.default_rng(2)
    w = rng.choice(np.array([-1, 0, 1], np.int8), (16, 512))
    assert np.array_equal(hf_to_tmm(pack_hf_bitnet(w)), pack_ternary_mxu(w, 128))


def test_fake_quant8_is_per_token_absmax():
    x = np.array([[0.5, -1.0, 0.25], [4.0, 2.0, -8.0]], np.float32)
    q = fake_quant8(x)
    assert np.abs(q - x).max() <= np.abs(x).max(axis=-1).max() / 127 / 2 + 1e-6
    assert q[0, 1] == -1.0 and q[1, 2] == -8.0


@pytest.mark.parametrize("scale", [1.0, 1.21875])
def test_tmm_on_repacked_hf_weights_tracks_the_hf_reference(scale):
    cfg = Config(S=1, D=128, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=16)
    D, M, N, K = cfg.D, 3, 8, 512
    rng = np.random.default_rng(3)
    x = rng.standard_normal((M, K), dtype=np.float32)
    w = rng.choice(np.array([-1, 0, 1], np.int8), (N, K))
    hf = pack_hf_bitnet(w)
    tmm_w = hf_to_tmm(hf, D)
    weight_at, scale_at, out_at = 0x20000, 0x30000, 4096
    img = np.zeros(cfg.DRAM_BYTES, np.uint8)
    img[:x.nbytes] = x.view(np.uint8).reshape(-1)
    img[weight_at:weight_at + tmm_w.size] = tmm_w.reshape(-1)
    sv = np.full(N, scale, np.float32)
    img[scale_at:scale_at + sv.nbytes] = sv.view(np.uint8)
    prog = [I.ld(0, 0, M * K), I.qact(0, M, 0, K // D, K),
            I.tmm(weight_at, scale_at, out_at, N, K // D, tmm_w.shape[1], N, M, 0, 4, rsc=True),
            I.st(0x40000, out_at, M * N), I.halt()]
    m = Machine(cfg, [prog], [img]).run().slices[0]
    got = m.dram[0x40000:0x40000 + 4 * M * N].view("<f4").reshape(M, N)
    want = reference_linear(x, hf, scale)
    err = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert err < 0.02
