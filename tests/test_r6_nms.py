import numpy as np
import pytest
from dataclasses import replace

from opentpu import Config, rtlsim, isa as I
from opentpu.isasim import Machine, SimError
from opentpu.kernels.nms_fpga import program, yolox_preprocess


def nms_case():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 9, 9], [1, 1, 9, 9],
                      [20, 20, 22, 22], [0, 0, 0, 0], [19, 19, 23, 23],
                      [30, 30, 35, 35], [2, 2, 11, 11]], np.float32)
    scores = np.array([.95, .9, .85, .8, .7, .6, .5, .4], np.float32)
    labels = np.array([0, 0, 1, 0, 1, 0, 1, 0], np.float32)
    cfg = Config(VCMP=True)
    image = np.zeros(cfg.DRAM_BYTES, np.uint8)
    vectors = np.concatenate((boxes.T.reshape(-1), scores, labels))
    image[:vectors.nbytes] = vectors.view(np.uint8)
    commands = program(len(scores), 7, 0, 0x10000, tmem_words=cfg.TMEM_WORDS,
                       score_threshold=.45, iou_threshold=.5)
    return cfg, image, commands, boxes, scores, labels


def reference(boxes, scores, labels, count):
    scores = scores.copy()
    out = []
    for _ in range(count):
        idx = int(np.argmax(scores))
        out.append([*boxes[idx], scores[idx], labels[idx], float(scores[idx] > .45)])
        lo = np.maximum(boxes[:, :2], boxes[idx, :2])
        hi = np.minimum(boxes[:, 2:], boxes[idx, 2:])
        wh = np.maximum(hi - lo, 0)
        intersection = wh[:, 0] * wh[:, 1]
        wh_all = np.maximum(boxes[:, 2:] - boxes[:, :2], 0)
        area = wh_all[:, 0] * wh_all[:, 1]
        union = area + area[idx] - intersection
        scores[(intersection > np.float32(.5) * union) & (labels == labels[idx])] = 0
        scores[idx] = 0
    return np.asarray(out, np.float32)


def test_device_nms_against_reference():
    cfg, image, commands, boxes, scores, labels = nms_case()
    sim = Machine(cfg, [commands], [image.copy()]).run().slices[0]
    got = sim.dram[0x10000:0x10000 + 7 * 7 * 4].view(np.float32).reshape(7, 7)
    np.testing.assert_array_equal(got, reference(boxes, scores, labels, 7))
    with pytest.raises(SimError, match="Config.VCMP"):
        Machine(Config(), [commands], [image.copy()]).run()
    with pytest.raises(ValueError, match="instruction memory"):
        program(1000, 1000, 0, 0x10000, tmem_words=65536)


def test_device_nms_rtl(have_verilator):
    cfg, image, commands, *_ = nms_case()
    expected = Machine(cfg, [commands], [image.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [commands], [image.copy()])
    np.testing.assert_array_equal(drams[0], expected.slices[0].dram)
    np.testing.assert_array_equal(tmems[0], expected.slices[0].tmem)


def test_device_nms_waitc_rtl(have_verilator):
    cfg, image, commands, *_ = nms_case()
    cfg = replace(cfg, D=128)
    expected = Machine(cfg, [commands], [image.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [commands], [image.copy()],
                                  uarch={"SEQ_WC": 1, "WIN": 16}, axi=True, stall=25, seed=719)
    np.testing.assert_array_equal(drams[0], expected.slices[0].dram)
    np.testing.assert_array_equal(tmems[0], expected.slices[0].tmem)


def yolox_case():
    cfg, image, _, boxes, scores, labels = nms_case()
    n, classes = len(boxes), 3
    xy = (boxes[:, :2] + boxes[:, 2:]) * np.float32(.5)
    wh = boxes[:, 2:] - boxes[:, :2]
    decoded = np.zeros((n, 5 + classes), np.float32)
    decoded[:, :4] = np.concatenate((xy, wh), axis=1)
    decoded[:, 4] = scores
    decoded[np.arange(n), 5 + labels.astype(int)] = 1
    image[:] = 0
    image[:decoded.nbytes] = decoded.view(np.uint8).reshape(-1)
    prep = yolox_preprocess(n, decoded.shape[1], 0, 0x4000,
                            tmem_words=cfg.TMEM_WORDS, chunk_rows=3)
    nms = program(n, 7, 0x4000, 0x10000, tmem_words=cfg.TMEM_WORDS,
                  score_threshold=.45, iou_threshold=.5)
    return cfg, image, prep + nms, boxes, scores, labels


def test_yolox_decode_to_nms_vectors_on_device():
    cfg, image, commands, boxes, scores, labels = yolox_case()
    sim = Machine(cfg, [commands], [image.copy()]).run().slices[0]
    got = sim.dram[0x10000:0x10000 + 7 * 7 * 4].view(np.float32).reshape(7, 7)
    np.testing.assert_array_equal(got, reference(boxes, scores, labels, 7))


def test_yolox_preprocessing_and_nms_rtl(have_verilator):
    cfg, image, commands, *_ = yolox_case()
    sim = Machine(cfg, [commands], [image.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [commands], [image.copy()])
    np.testing.assert_array_equal(drams[0], sim.slices[0].dram)
    np.testing.assert_array_equal(tmems[0], sim.slices[0].tmem)


@pytest.mark.parametrize("seed", [0, 4, 19])
def test_random_boxes_against_reference(seed):
    rng = np.random.default_rng(seed)
    n, count = 24, 12
    origin = rng.uniform(0, 20, (n, 2)).astype(np.float32)
    size = rng.uniform(.5, 10, (n, 2)).astype(np.float32)
    boxes = np.concatenate((origin, origin + size), axis=1)
    scores = rng.uniform(.01, .99, n).astype(np.float32)
    labels = rng.integers(0, 3, n).astype(np.float32)
    cfg = Config(VCMP=True)
    image = np.zeros(cfg.DRAM_BYTES, np.uint8)
    vectors = np.concatenate((boxes.T.reshape(-1), scores, labels))
    image[:vectors.nbytes] = vectors.view(np.uint8)
    commands = program(n, count, 0, 0x10000, tmem_words=cfg.TMEM_WORDS,
                       score_threshold=.45, iou_threshold=.5)
    sim = Machine(cfg, [commands], [image]).run().slices[0]
    got = sim.dram[0x10000:0x10000 + 28 * count].view(np.float32).reshape(count, 7)
    np.testing.assert_array_equal(got, reference(boxes, scores, labels, count))
