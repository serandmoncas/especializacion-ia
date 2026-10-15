"""Métricas de clasificación binaria desde cero. `y_true` ∈ {0, 1}; `scores`: puntaje o prob."""

import numpy as np


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    y_true, y_pred = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool)
    return {
        "tp": int(np.sum(y_true & y_pred)),
        "fp": int(np.sum(~y_true & y_pred)),
        "fn": int(np.sum(y_true & ~y_pred)),
        "tn": int(np.sum(~y_true & ~y_pred)),
    }


def precision_recall_f1(y_true, y_pred) -> tuple[float, float, float]:
    c = confusion_counts(y_true, y_pred)
    precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _sorted_counts(y_true, scores):
    """TP y FP acumulados al bajar el umbral, colapsando puntajes empatados en un solo punto."""
    y_true, scores = np.asarray(y_true, dtype=float), np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    y, s = y_true[order], scores[order]
    distinct = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tps = np.cumsum(y)[distinct]
    fps = (distinct + 1) - tps
    return tps, fps, s[distinct]


def roc_curve(y_true, scores) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(FPR, TPR, umbrales), comenzando en (0, 0)."""
    tps, fps, thr = _sorted_counts(y_true, scores)
    tpr = np.r_[0, tps / tps[-1]]
    fpr = np.r_[0, fps / fps[-1]]
    return fpr, tpr, np.r_[np.inf, thr]


def roc_auc(y_true, scores) -> float:
    """AUC = P(puntaje de un positivo > puntaje de un negativo), empates cuentan 1/2.

    Se calcula con rangos (estadístico U de Mann–Whitney) en O(n log n), sin construir la curva.
    """
    from scipy.stats import rankdata

    y_true = np.asarray(y_true).astype(bool)
    ranks = rankdata(scores)  # rango promedio en empates
    n_pos, n_neg = y_true.sum(), (~y_true).sum()
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def precision_recall_curve(y_true, scores) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(precisión, recall, umbrales) en orden de umbral decreciente."""
    tps, fps, thr = _sorted_counts(y_true, scores)
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return precision, recall, thr


def average_precision(y_true, scores) -> float:
    """AP = Σ (Rₙ − Rₙ₋₁) Pₙ: área bajo la curva PR sin interpolar (como sklearn)."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return float(np.sum(np.diff(np.r_[0, recall]) * precision))


def log_loss(y_true, proba, eps: float = 1e-15) -> float:
    y, p = np.asarray(y_true, dtype=float), np.clip(np.asarray(proba, dtype=float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_score(y_true, proba) -> float:
    return float(np.mean((np.asarray(proba, dtype=float) - np.asarray(y_true, dtype=float)) ** 2))


def calibration_bins(y_true, proba, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(probabilidad media predicha, frecuencia observada, tamaño) por bin de ancho uniforme.

    Bins vacíos se omiten. Un modelo calibrado tiene ambas columnas iguales.
    """
    y, p = np.asarray(y_true, dtype=float), np.asarray(proba, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    count = np.bincount(bins, minlength=n_bins)
    mean_p = np.bincount(bins, weights=p, minlength=n_bins)
    freq = np.bincount(bins, weights=y, minlength=n_bins)
    keep = count > 0
    return mean_p[keep] / count[keep], freq[keep] / count[keep], count[keep]


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    """ECE = Σ_b (n_b/n) |frecuencia_b − confianza_b|."""
    mean_p, freq, count = calibration_bins(y_true, proba, n_bins)
    return float(np.sum(count * np.abs(freq - mean_p)) / count.sum())
