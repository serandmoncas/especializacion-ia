"""Solución ejercicio 1: k-NN y concentración de distancias.

En alta dimensión las distancias se concentran: la diferencia entre el vecino más cercano y el
más lejano se vuelve pequeña frente a la distancia misma, y “cercano” deja de discriminar.
Por eso k-NN (y todo método basado en distancias) necesita features relevantes y escaladas,
o reducir dimensión antes.
"""

import numpy as np


class KNNClassifier:
    def __init__(self, k: int = 5) -> None:
        self.k = k

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.classes_, self.y_ = np.unique(y, return_inverse=True)
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        d2 = (X**2).sum(1)[:, None] - 2 * X @ self.X_.T + (self.X_**2).sum(1)[None, :]
        nearest = np.argpartition(d2, self.k - 1, axis=1)[
            :, : self.k
        ]  # O(n) por fila, sin ordenar todo
        votes = np.zeros((len(X), len(self.classes_)))
        np.add.at(votes, (np.arange(len(X))[:, None], self.y_[nearest]), 1)
        return self.classes_[votes.argmax(axis=1)]


def distance_contrast(d: int, n: int = 1000, seed: int = 0) -> float:
    """(max − min)/min de las distancias de un punto de consulta a n puntos uniformes en [0,1]^d."""
    rng = np.random.default_rng(seed)
    pts, q = rng.random((n, d)), rng.random(d)
    dist = np.linalg.norm(pts - q, axis=1)
    return float((dist.max() - dist.min()) / dist.min())


def check() -> None:
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    Xtr, Xte, ytr, _ = train_test_split(X, y, random_state=0)
    ours = KNNClassifier(7).fit(Xtr, ytr).predict(Xte)
    ref = KNeighborsClassifier(7).fit(Xtr, ytr).predict(Xte)
    assert np.mean(ours == ref) > 0.99, "las predicciones deberían coincidir (salvo empates)"
    contrasts = {d: distance_contrast(d) for d in (2, 10, 100, 1000)}
    assert contrasts[2] > 10 * contrasts[1000]
    print(
        "ej01 ok — contraste (max−min)/min por dimensión:",
        {d: round(c, 3) for d, c in contrasts.items()},
    )


if __name__ == "__main__":
    check()
