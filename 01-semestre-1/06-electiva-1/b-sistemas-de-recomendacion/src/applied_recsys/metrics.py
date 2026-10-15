"""Métricas de recomendación: error de rating y calidad del ranking top-N."""

import numpy as np
from scipy import sparse


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def top_n(scores: np.ndarray, exclude: sparse.csr_matrix, n: int) -> np.ndarray:
    """Los n ítems de mayor puntaje por fila, excluyendo los ya vistos en train (−∞).

    argpartition (O(I)) y luego ordenar solo los n elegidos, en vez de ordenar todo el catálogo.
    """
    scores = np.array(scores, dtype=np.float64, copy=True)
    coo = exclude.tocoo()
    scores[coo.row, coo.col] = -np.inf
    part = np.argpartition(-scores, n, axis=1)[:, :n]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def ranking_metrics(recs: np.ndarray, relevant: list[set[int]], k: int) -> dict[str, np.ndarray]:
    """Métricas por usuario (arreglos, para poder hacer IC bootstrap sobre usuarios).

    - precision@k: fracción de las k recomendaciones que son relevantes.
    - recall@k: fracción de los relevantes que aparecen en las k.
    - ndcg@k: ganancia descontada por posición (1/log₂(pos + 1)), normalizada por la ideal.
    - hit@k: ¿al menos un relevante en las k?
    """
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    out = {name: np.zeros(len(relevant)) for name in ("precision", "recall", "ndcg", "hit")}
    for row, rel in enumerate(relevant):
        hits = np.array([item in rel for item in recs[row, :k]], dtype=float)
        out["precision"][row] = hits.mean()
        out["recall"][row] = hits.sum() / len(rel)
        ideal = discounts[: min(len(rel), k)].sum()
        out["ndcg"][row] = (hits * discounts).sum() / ideal
        out["hit"][row] = float(hits.any())
    return out


def catalog_coverage(recs: np.ndarray, n_items: int) -> float:
    """Fracción del catálogo que aparece en alguna lista recomendada."""
    return len(np.unique(recs)) / n_items


def mean_popularity_percentile(recs: np.ndarray, popularity: np.ndarray) -> float:
    """Percentil de popularidad medio de lo recomendado (100 = lo más popular)."""
    pct = popularity.argsort().argsort() / (len(popularity) - 1) * 100
    return float(pct[recs].mean())
