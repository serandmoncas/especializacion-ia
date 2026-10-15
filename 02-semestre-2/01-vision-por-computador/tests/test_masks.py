import numpy as np
import pytest
from applied_cv.masks import dice, mask_iou


def test_dice_and_iou_known_masks():
    a = np.zeros((4, 4), bool)
    b = np.zeros((4, 4), bool)
    a[:2] = True  # 8 pixeles
    b[1:3] = True  # 8 pixeles, 4 en común
    assert dice(a, b) == pytest.approx(2 * 4 / 16)
    assert mask_iou(a, b) == pytest.approx(4 / 12)


def test_empty_masks_agree_perfectly():
    empty = np.zeros((3, 3), bool)
    assert dice(empty, empty) == 1.0 and mask_iou(empty, empty) == 1.0
    full = np.ones((3, 3), bool)
    assert dice(empty, full) == 0.0 and mask_iou(empty, full) == 0.0
