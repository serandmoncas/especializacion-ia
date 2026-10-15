import numpy as np
import pytest
from applied_cv.boxes import average_precision, average_precision_range, box_iou, nms

torch = pytest.importorskip("torch")
ops = pytest.importorskip("torchvision.ops")
RNG = np.random.default_rng(0)


def _random_boxes(n):
    xy = RNG.random((n, 2)) * 50
    return np.c_[xy, xy + RNG.random((n, 2)) * 30 + 1]


def test_box_iou_matches_torchvision():
    a, b = _random_boxes(7), _random_boxes(5)
    ref = ops.box_iou(torch.tensor(a), torch.tensor(b)).numpy()
    np.testing.assert_allclose(box_iou(a, b), ref, atol=1e-12)


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
def test_nms_matches_torchvision(threshold):
    boxes, scores = _random_boxes(40), RNG.random(40)
    ref = ops.nms(torch.tensor(boxes), torch.tensor(scores), threshold).numpy()
    np.testing.assert_array_equal(nms(boxes, scores, threshold), ref)


def test_average_precision_hand_computed():
    # 2 cajas reales; predicciones 0.9 (TP), 0.8 (FP), 0.7 (TP):
    # precisión 1, 1/2, 2/3; recall 1/2, 1/2, 1; envolvente 1, 2/3, 2/3 → AP = ½·1 + ½·⅔
    gt = [np.array([[0, 0, 10, 10], [20, 20, 30, 30]])]
    pred = [np.array([[0, 0, 10, 10], [50, 50, 60, 60], [20, 20, 30, 31]])]
    ap = average_precision(pred, [np.array([0.9, 0.8, 0.7])], gt)
    assert ap == pytest.approx(0.5 + 0.5 * 2 / 3)


def test_average_precision_duplicates_and_edge_cases():
    gt = [np.array([[0, 0, 10, 10], [20, 20, 30, 30]])]
    duplicate = [np.array([[0, 0, 10, 10], [0, 0, 10, 10]])]
    assert average_precision(duplicate, [np.array([0.9, 0.8])], gt) == pytest.approx(0.5)
    assert average_precision([gt[0]], [np.array([1.0, 0.5])], gt) == pytest.approx(1.0)
    assert average_precision_range([gt[0]], [np.array([1.0, 0.5])], gt) == pytest.approx(1.0)
    assert average_precision([np.zeros((0, 4))], [np.zeros(0)], gt) == 0.0
    assert np.isnan(average_precision([gt[0]], [np.array([1.0, 0.5])], [np.zeros((0, 4))]))
