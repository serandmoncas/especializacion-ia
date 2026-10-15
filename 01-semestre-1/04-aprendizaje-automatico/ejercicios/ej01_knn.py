"""Ejercicio 1: k-NN y la maldición de la dimensión. Ver README.md."""

import numpy as np


class KNNClassifier:
    def __init__(self, k: int = 5) -> None:
        self.k = k

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.classes_, self.y_ = np.unique(y, return_inverse=True)
        return self

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError


def distance_contrast(d: int, n: int = 1000, seed: int = 0) -> float:
    """(max − min)/min de las distancias de un punto de consulta a n puntos uniformes en [0,1]^d."""
    raise NotImplementedError


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
