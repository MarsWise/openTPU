"""The repetition penalty's tables over a prompt built on the device (kernels/penalty.py): the
host writes the ids and their count only; the tables equal the host's former np.unique ones."""
import numpy as np
import pytest

from opentpu import Config, rtlsim
from opentpu.isasim import Machine
from opentpu.kernels import penalty as P
from opentpu.llm import generate as G


def _case(V, ids, R, nmax=None):
    cfg = Config(D=128, MCOLS=8, ACT_BLOCKS=16, TMEM_WORDS=1 << 14, DRAM_BYTES=1 << 20)
    nmax = nmax or max(len(ids), 1)
    a = dict(ids=0x1000, count=0x0800, state=0x0400, pa=0x20000, pb=0x40000)
    img = np.zeros(cfg.DRAM_BYTES, np.uint8)
    img[0x20000:0x60000] = 0x77                          # stale tables (another request's)
    st = np.zeros(G.STATE_WORDS, np.float32)
    st[G.S_PEN], st[G.S_PENINV] = R, np.float32(1.0) / np.float32(R)
    img[a["state"]:a["state"] + 4 * G.STATE_WORDS] = st.view(np.uint8)
    w = np.full(nmax, 123456.0, np.float32)              # past the count: never read as ids
    w[:len(ids)] = ids
    img[a["ids"]:a["ids"] + 4 * nmax] = w.view(np.uint8)
    img[a["count"]:a["count"] + 4] = np.float32(len(ids)).reshape(1).view(np.uint8)
    prog = P.program(cfg, V=V, nmax=nmax, buf=1024, **a)
    return cfg, prog, img, a


@pytest.mark.parametrize("V,ids,R", [(1000, [3, 999, 3, 0, 500], 1.1), (2500, [], 1.3),
                                     (4096, list(range(0, 4096, 7)), 2.0)])
def test_tables_are_the_hosts(V, ids, R):
    cfg, prog, img, a = _case(V, ids, R, nmax=600)
    s = Machine(cfg, [prog], [img]).run().slices[0]
    pa = s.dram[a["pa"]:a["pa"] + 4 * V].copy().view(np.float32)
    pb = s.dram[a["pb"]:a["pb"] + 4 * V].copy().view(np.float32)
    wa, wb = P.reference(V, ids, R)
    np.testing.assert_array_equal(pa, wa)
    np.testing.assert_array_equal(pb, wb)
    assert (s.dram[a["pa"] + 4 * V:a["pa"] + 4 * V + 16] == 0x77).all()


def test_penalty_init_rtl(have_verilator):
    cfg, prog, img, a = _case(1500, [1, 1499, 77, 1, 640], 1.2, nmax=16)
    ref = Machine(cfg, [prog], [img.copy()]).run().slices[0]
    drams, tmems, _ = rtlsim.run(cfg, [prog], [img.copy()], axi=True, stall=35, seed=4)
    np.testing.assert_array_equal(drams[0], ref.dram)
    np.testing.assert_array_equal(tmems[0], ref.tmem)
