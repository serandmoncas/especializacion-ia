"""Métricas (F1, AP, recall@k, NDCG@k), umbral por F1, bootstrap por conglomerados y Holm."""

from collections.abc import Callable

import numpy as np


def f1_binary(y_true, y_pred) -> float:
    """F1 de la clase positiva; 0 si no hay verdaderos positivos."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int(np.sum(y_true & y_pred))
    if tp == 0:
        return 0.0
    precision = tp / int(np.sum(y_pred))
    recall = tp / int(np.sum(y_true))
    return 2 * precision * recall / (precision + recall)


def average_precision(y_true, scores) -> float:
    """AP como `sklearn.metrics.average_precision_score` (empates agrupados); 0 sin positivos."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if y_true.sum() == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    y, s = y_true[order], scores[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]  # último índice de cada umbral
    tps = np.cumsum(y)[last]
    precision = tps / (last + 1)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def best_threshold(y_true, scores) -> float:
    """Umbral (predecir `scores >= t`) con F1 máximo; en empate, el mayor. Sin positivos: +inf."""
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=float)
    if not y_true.any():
        return float("inf")
    best_t, best_f1 = float("inf"), -1.0
    for t in np.unique(scores)[::-1]:
        f1 = f1_binary(y_true, scores >= t)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("el conjunto relevante está vacío")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """NDCG binario; el ideal pone todos los relevantes (aunque no estén en `ranked`) arriba."""
    if not relevant:
        raise ValueError("el conjunto relevante está vacío")
    dcg = sum(1 / np.log2(i + 2) for i, r in enumerate(ranked[:k]) if r in relevant)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return float(dcg / ideal)


def cluster_bootstrap(
    clusters, stat: Callable[[np.ndarray], float], rng: np.random.Generator, n_boot: int = 2000
) -> dict:
    """Remuestrea conglomerados completos y recalcula `stat(índices)`.

    Devuelve la diferencia observada, el IC 95 % percentil y p unilateral (H1: Δ > 0)
    = max(fracción de réplicas ≤ 0, 1/n_boot). `_boots` queda para diagnóstico.
    """
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    members = {c: np.flatnonzero(clusters == c) for c in uniq}
    observed = float(stat(np.arange(len(clusters))))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        boots[i] = stat(np.concatenate([members[c] for c in pick]))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p = max(float(np.mean(boots <= 0)), 1 / n_boot)
    return {"delta": observed, "ci": [float(lo), float(hi)], "p": p, "_boots": boots}


def holm(pvalues) -> list[float]:
    """Corrección de Holm-Bonferroni (escalonada, monótona y acotada en 1)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()
