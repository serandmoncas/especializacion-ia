"""Cajas [x1, y1, x2, y2]: IoU, supresión de no máximos (NMS) y precisión promedio (AP)."""

import numpy as np


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matriz IoU [len(a), len(b)]: intersección sobre unión de cada par de cajas."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=-1)
    area_a = (a[:, 2:] - a[:, :2]).clip(0).prod(axis=-1)
    area_b = (b[:, 2:] - b[:, :2]).clip(0).prod(axis=-1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Índices conservados, de mayor a menor puntaje (mismo resultado que torchvision.ops.nms).

    Toma la caja de mayor puntaje, descarta las que se le superponen más que el umbral y repite: un
    detector propone muchas cajas casi iguales por objeto y NMS deja una.
    """
    boxes = np.asarray(boxes, dtype=float)
    order = np.argsort(-np.asarray(scores), kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        order = rest[box_iou(boxes[i], boxes[rest])[0] <= iou_threshold]
    return np.array(keep, dtype=int)


def average_precision(
    pred_boxes: list[np.ndarray],
    pred_scores: list[np.ndarray],
    gt_boxes: list[np.ndarray],
    iou_threshold: float = 0.5,
) -> float:
    """AP de una clase sobre varias imágenes: área bajo la envolvente de precisión-recall.

    Las predicciones de todas las imágenes se ordenan por puntaje. Cada una se empareja con la caja
    real libre de mayor IoU de su imagen; si ese IoU ≥ umbral es verdadero positivo y la caja real
    queda usada (una segunda detección del mismo objeto es falso positivo).
    """
    n_gt = sum(len(np.asarray(g).reshape(-1, 4)) for g in gt_boxes)
    if n_gt == 0:
        return float("nan")
    records = [
        (score, img, j)
        for img, scores in enumerate(pred_scores)
        for j, score in enumerate(np.asarray(scores))
    ]
    if not records:
        return 0.0
    records.sort(key=lambda r: -r[0])
    gts = [np.asarray(g, dtype=float).reshape(-1, 4) for g in gt_boxes]
    used = [np.zeros(len(g), dtype=bool) for g in gts]
    tp = np.zeros(len(records))
    for k, (_, img, j) in enumerate(records):
        if len(gts[img]) == 0:
            continue
        ious = box_iou(np.asarray(pred_boxes[img])[j], gts[img])[0]
        ious[used[img]] = -1.0
        best = int(np.argmax(ious))
        if ious[best] >= iou_threshold:
            used[img][best] = True
            tp[k] = 1.0
    hits = np.cumsum(tp)
    recall = hits / n_gt
    precision = hits / np.arange(1, len(records) + 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float((np.diff(np.r_[0.0, recall]) * envelope).sum())


def average_precision_range(
    pred_boxes: list[np.ndarray], pred_scores: list[np.ndarray], gt_boxes: list[np.ndarray]
) -> float:
    """AP50:95: promedio del AP con umbrales de IoU 0.50, 0.55, …, 0.95 (convención COCO)."""
    return float(
        np.mean(
            [
                average_precision(pred_boxes, pred_scores, gt_boxes, t)
                for t in np.linspace(0.5, 0.95, 10)
            ]
        )
    )
