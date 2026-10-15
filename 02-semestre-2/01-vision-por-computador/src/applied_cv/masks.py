"""Métricas de segmentación sobre máscaras booleanas."""

import numpy as np


def dice(pred: np.ndarray, target: np.ndarray) -> float:
    """2·|A∩B| / (|A| + |B|). Dos máscaras vacías coinciden perfectamente (1.0)."""
    pred, target = np.asarray(pred, dtype=bool), np.asarray(target, dtype=bool)
    total = pred.sum() + target.sum()
    return 1.0 if total == 0 else float(2 * (pred & target).sum() / total)


def mask_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """|A∩B| / |A∪B|. Siempre ≤ Dice: IoU = Dice / (2 − Dice)."""
    pred, target = np.asarray(pred, dtype=bool), np.asarray(target, dtype=bool)
    union = (pred | target).sum()
    return 1.0 if union == 0 else float((pred & target).sum() / union)
