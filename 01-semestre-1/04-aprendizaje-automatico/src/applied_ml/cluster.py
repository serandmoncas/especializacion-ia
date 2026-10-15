"""K-means (algoritmo de Lloyd) con inicialización k-means++."""

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


def _sq_dist(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Distancias euclidianas al cuadrado [n, k] vía ‖x‖² − 2x·c + ‖c‖² (sin bucles)."""
    d = (X**2).sum(1)[:, None] - 2 * X @ C.T + (C**2).sum(1)[None, :]
    return np.maximum(d, 0)


def kmeans_plusplus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Elige cada nuevo centro con probabilidad ∝ D(x)², la distancia al centro más cercano.

    Garantiza en esperanza una inercia O(log k) veces la óptima (Arthur & Vassilvitskii, 2007).
    """
    centers = [X[rng.integers(len(X))]]
    d2 = _sq_dist(X, np.array(centers))[:, 0]
    for _ in range(1, k):
        probs = d2 / d2.sum() if d2.sum() > 0 else None
        centers.append(X[rng.choice(len(X), p=probs)])
        d2 = np.minimum(d2, _sq_dist(X, centers[-1][None, :])[:, 0])
    return np.array(centers)


class KMeans(ClusterMixin, BaseEstimator):
    """Minimiza la inercia Σ‖x − μ_c(x)‖² alternando asignación y actualización de centros.

    Cada paso no aumenta la inercia, así que converge, pero a un óptimo *local*: por eso se
    repite `n_init` veces con distintas semillas y se queda la mejor.
    """

    def __init__(
        self, n_clusters=8, init="k-means++", n_init=10, max_iter=300, tol=1e-6, random_state=None
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _single_run(self, X: np.ndarray, rng: np.random.Generator):
        if self.init == "k-means++":
            C = kmeans_plusplus(X, self.n_clusters, rng)
        elif self.init == "random":
            C = X[rng.choice(len(X), self.n_clusters, replace=False)]
        else:
            raise ValueError(f"init desconocido: {self.init!r}")
        n_iter = 0
        for _ in range(self.max_iter):
            n_iter += 1
            labels = _sq_dist(X, C).argmin(axis=1)
            new_C = np.array(
                [
                    X[labels == j].mean(axis=0) if np.any(labels == j) else C[j]
                    for j in range(self.n_clusters)
                ]
            )
            shift = np.sum((new_C - C) ** 2)
            C = new_C
            if shift <= self.tol:
                break
        d = _sq_dist(X, C)
        labels = d.argmin(axis=1)
        return C, labels, float(d[np.arange(len(X)), labels].sum()), n_iter

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            run = self._single_run(X, rng)
            if best is None or run[2] < best[2]:
                best = run
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def predict(self, X) -> np.ndarray:
        return _sq_dist(np.asarray(X, dtype=float), self.cluster_centers_).argmin(axis=1)
