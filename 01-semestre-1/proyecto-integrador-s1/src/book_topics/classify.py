"""Libro → temas: prevalencia, k vecinos y TF-IDF + logística calibrada, con umbrales por tema."""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier


def book_text(titles: pd.Series, descriptions: pd.Series) -> pd.Series:
    return titles.astype(str) + ". " + descriptions.reindex(titles.index).fillna("").astype(str)


class PrevalenceModel:
    def fit(self, Y: np.ndarray) -> "PrevalenceModel":
        self.prevalence_ = np.asarray(Y, dtype=float).mean(axis=0)
        return self

    def predict_proba(self, n: int) -> np.ndarray:
        return np.tile(self.prevalence_, (n, 1))


class KNNTopics:
    """Temas = promedio de los temas de los k libros más parecidos, ponderado por coseno."""

    def __init__(self, k: int = 20):
        self.k = k

    def fit(self, X, Y: np.ndarray) -> "KNNTopics":
        self.X_, self.Y_ = X, np.asarray(Y, dtype=float)
        return self

    def predict_proba(self, X) -> np.ndarray:
        sim = (X @ self.X_.T).toarray()  # TF-IDF con norma L2: producto punto = coseno
        k = min(self.k, sim.shape[1])
        idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        w = np.take_along_axis(sim, idx, axis=1)
        votes = np.einsum("nk,nkt->nt", w, self.Y_[idx])
        total = w.sum(axis=1, keepdims=True)
        return np.where(total > 0, votes / np.where(total > 0, total, 1), self.Y_.mean(axis=0))


class TextTopicModel:
    """TF-IDF (1–2 gramas, sublineal) + logística uno-contra-todos con calibración sigmoide."""

    def __init__(
        self, C: float = 4.0, max_features: int = 50_000, min_df: int = 2, calibration_cv: int = 3
    ):
        self.C, self.max_features, self.min_df, self.calibration_cv = (
            C,
            max_features,
            min_df,
            calibration_cv,
        )

    def fit(self, texts: list[str], Y: np.ndarray) -> "TextTopicModel":
        self.vectorizer_ = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=self.min_df,
            max_features=self.max_features,
            stop_words="english",
        )
        X = self.vectorizer_.fit_transform(texts)
        base = CalibratedClassifierCV(
            LogisticRegression(C=self.C, max_iter=2000), method="sigmoid", cv=self.calibration_cv
        )
        self.model_ = OneVsRestClassifier(base, n_jobs=-1).fit(X, np.asarray(Y, dtype=int))
        self.prevalence_ = np.asarray(Y, dtype=float).mean(axis=0)
        return self

    def transform(self, texts: list[str]):
        return self.vectorizer_.transform(texts)

    def low_confidence(self, texts: list[str]) -> np.ndarray:
        """True si el texto no comparte vocabulario con el entrenamiento (vector TF-IDF nulo)."""
        return np.asarray(self.transform(texts).getnnz(axis=1) == 0)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = self.transform(texts)
        P = self.model_.predict_proba(X)
        P[np.asarray(X.getnnz(axis=1) == 0)] = self.prevalence_  # sin evidencia: la prevalencia
        return P


def fit_thresholds(
    Y_val: np.ndarray, P_val: np.ndarray, grid: np.ndarray | None = None
) -> np.ndarray:
    grid = np.linspace(0.05, 0.95, 19) if grid is None else grid
    return np.array(
        [
            grid[
                int(
                    np.argmax(
                        [f1_score(Y_val[:, t], P_val[:, t] >= g, zero_division=0) for g in grid]
                    )
                )
            ]
            for t in range(P_val.shape[1])
        ]
    )


def apply_thresholds(P: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    pred = thresholds <= P
    none = ~pred.any(axis=1)
    pred[np.flatnonzero(none), np.argmax(P[none] / thresholds, axis=1)] = True
    return pred
