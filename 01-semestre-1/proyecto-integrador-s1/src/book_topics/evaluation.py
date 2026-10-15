"""Métricas multi-etiqueta y bootstrap por conglomerados para comparaciones pareadas."""

from collections.abc import Callable

import numpy as np
from sklearn.metrics import f1_score


def multilabel_report(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    return {
        "micro_f1": float(f1_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "per_topic_f1": f1_score(Y_true, Y_pred, average=None, zero_division=0),
    }


def precision_at_k(Y_true: np.ndarray, proba: np.ndarray, k: int = 3) -> float:
    top = np.argsort(-proba, axis=1)[:, :k]
    return float(np.take_along_axis(np.asarray(Y_true, bool), top, axis=1).mean())


def cluster_bootstrap_diff(
    stat_a: Callable[[np.ndarray], float],
    stat_b: Callable[[np.ndarray], float],
    groups: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """IC 95 % y p-valor bilateral de stat_a − stat_b remuestreando *conglomerados* completos.

    Las observaciones de un mismo grupo (libros de un autor, recomendaciones a un lector) están
    correlacionadas: remuestrearlas sueltas subestima la varianza (intervalos falsamente estrechos).
    El p-valor tiene resolución 1/n_boot: el informe muestra “< 1/n_boot” cuando toca ese piso.
    """
    rng = np.random.default_rng(seed)
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    n_groups = inverse.max() + 1
    members = [np.flatnonzero(inverse == g) for g in range(n_groups)]
    everything = np.arange(len(inverse))
    observed = stat_a(everything) - stat_b(everything)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, n_groups, n_groups)])
        diffs[k] = stat_a(idx) - stat_b(idx)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return {
        "diff": float(observed),
        "ci": (float(lo), float(hi)),
        "p_value": float(min(1.0, max(p, 1 / n_boot))),
    }


def per_user_diff_bootstrap(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> dict:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return cluster_bootstrap_diff(
        lambda i: a[i].mean(), lambda i: b[i].mean(), np.arange(len(a)), n_boot, seed
    )
