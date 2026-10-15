"""Embeddings de concepto → rasgos de par asimétricos → regresión logística."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from research_paths.evaluation import average_precision


def pair_features(ea: np.ndarray, eb: np.ndarray) -> np.ndarray:
    """[a, b, a − b, a·b]: el orden del par importa (a−b cambia de signo al invertirlo)."""
    return np.hstack([ea, eb, ea - eb, ea * eb])


def fit_pair_logreg(X: np.ndarray, y: np.ndarray, C: float) -> LogisticRegression:
    return LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(X, y)


def select_c(X_tr, y_tr, X_val, y_val, grid=(0.01, 0.1, 1.0, 10.0)) -> float:
    """C con mayor AP en validación (en empate, el menor: más regularizado)."""
    scores = [
        average_precision(y_val, fit_pair_logreg(X_tr, y_tr, c).predict_proba(X_val)[:, 1])
        for c in grid
    ]
    return float(grid[int(np.argmax(scores))])
