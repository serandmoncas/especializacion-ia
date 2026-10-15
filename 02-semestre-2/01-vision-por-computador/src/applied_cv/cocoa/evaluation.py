"""Métricas del proyecto y estadística: umbral a FPR fija, bootstrap por ráfaga y Holm."""

from collections.abc import Callable

import numpy as np
from sklearn.metrics import f1_score

from applied_cv.boxes import box_iou


def sick_score(proba: np.ndarray) -> np.ndarray:
    """P(enferma) = P(m1) + P(m2) + P(m3) = 1 − P(h0)."""
    return 1.0 - np.asarray(proba)[:, 0]


def threshold_at_fpr(negative_scores: np.ndarray, fpr: float = 0.1) -> float:
    """Umbral tal que, en las sanas dadas (validación), la fracción con puntaje > umbral es ≤ fpr.

    Sin sanas (solo pasa con datos de prueba diminutos) devuelve 0.5.
    """
    negative_scores = np.asarray(negative_scores, dtype=float)
    if negative_scores.size == 0:
        return 0.5
    return float(np.quantile(negative_scores, 1 - fpr, method="higher"))


def recall_above(scores: np.ndarray, threshold: float) -> float:
    """Fracción de puntajes > umbral (recall si son enfermas, FPR si son sanas)."""
    scores = np.asarray(scores, dtype=float)
    return float(np.mean(scores > threshold)) if scores.size else float("nan")


def macro_f1(y_true: np.ndarray, proba: np.ndarray, n_classes: int = 4) -> float:
    return float(
        f1_score(
            y_true,
            np.asarray(proba).argmax(axis=1),
            average="macro",
            labels=list(range(n_classes)),
            zero_division=0,
        )
    )


def cluster_members(clusters: np.ndarray) -> list[np.ndarray]:
    _, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    return [np.flatnonzero(inverse == k) for k in range(inverse.max() + 1)]


def resample(members: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Remuestrea ráfagas enteras con reemplazo: sus fotos no son independientes entre sí."""
    chosen = rng.integers(0, len(members), len(members))
    return np.concatenate([members[k] for k in chosen])


def _summary(observed: float, dist: np.ndarray) -> dict:
    dist = dist[~np.isnan(dist)]
    lo, hi = np.percentile(dist, [2.5, 97.5])
    p = 2 * min(np.mean(dist <= 0), np.mean(dist >= 0))
    return {
        "diff": float(observed),
        "ci": [float(lo), float(hi)],
        "p_value": float(min(1.0, max(p, 1 / len(dist)))),  # resolución 1/n_boot
        "n_boot": len(dist),
    }


def bootstrap_diff(
    stat: Callable[[np.ndarray], float], clusters: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> dict:
    """IC y p-valor bilateral de un estadístico de diferencia, remuestreando conglomerados."""
    members, rng = cluster_members(clusters), np.random.default_rng(seed)
    observed = stat(np.arange(len(clusters)))
    return _summary(observed, np.array([stat(resample(members, rng)) for _ in range(n_boot)]))


def bootstrap_unpaired(
    stat_a: Callable[[np.ndarray], float],
    clusters_a: np.ndarray,
    stat_b: Callable[[np.ndarray], float],
    clusters_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Diferencia stat_a − stat_b entre dos conjuntos distintos, remuestreados por separado."""
    rng = np.random.default_rng(seed)
    ma, mb = cluster_members(clusters_a), cluster_members(clusters_b)
    observed = stat_a(np.arange(len(clusters_a))) - stat_b(np.arange(len(clusters_b)))
    dist = np.array([stat_a(resample(ma, rng)) - stat_b(resample(mb, rng)) for _ in range(n_boot)])
    return _summary(observed, dist)


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Corrección de Holm-Bonferroni (controla el error familiar sobre todas las hipótesis)."""
    names = sorted(p_values, key=p_values.get)
    adjusted, running = {}, 0.0
    for k, name in enumerate(names):
        running = max(running, min(1.0, (len(names) - k) * p_values[name]))
        adjusted[name] = running
    return adjusted


def decide(result: dict, p_holm: float, alpha: float = 0.05) -> dict:
    """Rechazar H0 no basta: la hipótesis se apoya solo si además la diferencia es positiva."""
    rejects = p_holm < alpha
    return {
        **result,
        "p_holm": float(p_holm),
        "rechaza_h0": bool(rejects),
        "apoya_hipotesis": bool(rejects and result["diff"] > 0),
    }


def match_boxes(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray, iou_threshold: float = 0.5
) -> np.ndarray:
    """Por caja real, la predicción emparejada (−1 si ninguna); uno a uno, por IoU descendente."""
    pred = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    gt = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    match = np.full(len(gt), -1)
    if len(pred) == 0 or len(gt) == 0:
        return match
    ious = box_iou(gt, pred)
    pairs = sorted(
        ((ious[i, j], i, j) for i, j in zip(*np.nonzero(ious >= iou_threshold), strict=True)),
        reverse=True,
    )
    used = set()
    for _, i, j in pairs:
        if match[i] == -1 and j not in used:
            match[i] = j
            used.add(j)
    return match
