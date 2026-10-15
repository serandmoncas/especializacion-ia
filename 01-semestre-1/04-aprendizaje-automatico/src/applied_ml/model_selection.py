"""Validación: k-fold estratificado, validación cruzada y bootstrap de métricas en test."""

from collections.abc import Callable

import numpy as np
from sklearn.base import clone


def stratified_kfold(
    y, n_splits: int = 5, rng: np.random.Generator | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Particiones (train, test) que preservan la proporción de cada clase en cada fold.

    Con clases desbalanceadas, un k-fold simple puede dejar folds casi sin positivos y
    disparar la varianza de la estimación.
    """
    rng = rng or np.random.default_rng(0)
    y = np.asarray(y)
    fold_of = np.empty(len(y), dtype=int)
    for cls in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == cls))
        fold_of[idx] = np.arange(len(idx)) % n_splits  # reparto round-robin dentro de la clase
    all_idx = np.arange(len(y))
    return [(all_idx[fold_of != k], all_idx[fold_of == k]) for k in range(n_splits)]


def cross_validate(
    estimator,
    X,
    y,
    scorer: Callable[[object, np.ndarray, np.ndarray], float],
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Clona y entrena el estimador en cada fold; devuelve el puntaje en cada fold de test.

    Todo lo que “aprende” de los datos (escalado, imputación, selección de features) debe vivir
    *dentro* del estimador (un Pipeline) para ajustarse solo con el fold de entrenamiento.
    """
    X, y = np.asarray(X), np.asarray(y)
    scores = []
    for train, test in folds:
        model = clone(estimator).fit(X[train], y[train])
        scores.append(scorer(model, X[test], y[test]))
    return np.array(scores)


def bootstrap_metric_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true,
    scores,
    *,
    n_boot: int = 2000,
    conf: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """(valor, IC bajo, IC alto) de una métrica en test, remuestreando pares (y, score).

    El conjunto de test es una muestra: su AUC tiene incertidumbre que conviene reportar.
    """
    rng = rng or np.random.default_rng(0)
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(metric(y_true[idx], scores[idx]))
    alpha = 1 - conf
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(metric(y_true, scores)), float(lo), float(hi)
