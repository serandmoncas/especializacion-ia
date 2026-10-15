"""Filtrado colaborativo basado en ítems (item-kNN)."""

import numpy as np
from scipy import sparse


def item_similarity(X: sparse.csr_matrix, *, shrink: float = 0.0) -> np.ndarray:
    """Similitud coseno entre columnas de X (ítems), con encogimiento.

    s_ij = (x_i · x_j) / (‖x_i‖‖x_j‖ + shrink). El término `shrink` castiga similitudes estimadas
    con pocos usuarios en común (Koren, 2008): dos películas vistas por un único usuario quedan con
    coseno 1 sin él.
    """
    X = sparse.csc_matrix(X, dtype=np.float64)
    dot = (X.T @ X).toarray()
    norms = np.sqrt(np.diag(dot))
    sim = dot / (np.outer(norms, norms) + shrink + 1e-12)
    np.fill_diagonal(sim, 0.0)
    return sim


def keep_top_k(sim: np.ndarray, k: int) -> np.ndarray:
    """Conserva en cada columna los k vecinos más similares (el resto a 0)."""
    if k >= sim.shape[0] - 1:
        return sim
    idx = np.argpartition(-sim, k, axis=0)[:k]
    out = np.zeros_like(sim)
    cols = np.arange(sim.shape[1])
    out[idx, cols] = sim[idx, cols]
    return out


class ItemKNN:
    """Item-kNN sobre residuos de un modelo de sesgos (coseno “ajustado”).

    Predicción de rating: r̂_ui = b_ui + Σ_{j∈N_k(i)∩R(u)} s_ij (r_uj − b_uj) / (Σ |s_ij| + λ).
    Puntaje de ranking: Σ_{j∈R(u)} s_ij (cuánto se parece i a lo que u ya vio).

    λ (`denom_reg`) importa: sin él, si u calificó un único vecino de i con similitud 0.01, el
    ajuste es el residuo completo de ese vecino. Con datos dispersos (pocos vecinos calificados por
    par) eso amplifica ruido hasta empeorar al modelo de sesgos. λ encoge el ajuste hacia 0 cuando
    la evidencia es débil.
    """

    def __init__(self, k: int = 50, shrink: float = 10.0, denom_reg: float = 0.5):
        self.k, self.shrink, self.denom_reg = k, shrink, denom_reg

    def fit(self, R: sparse.csr_matrix, bias_model) -> "ItemKNN":
        coo = R.tocoo()
        resid = coo.data - bias_model.predict(coo.row, coo.col)
        self.resid_ = sparse.csr_matrix((resid, (coo.row, coo.col)), shape=R.shape)
        self.bias_ = bias_model
        self.sim_ = keep_top_k(item_similarity(self.resid_, shrink=self.shrink), self.k)
        self.binary_ = sparse.csr_matrix((np.ones_like(resid), (coo.row, coo.col)), shape=R.shape)
        return self

    def predict(self, u: np.ndarray, i: np.ndarray) -> np.ndarray:
        num = np.asarray(self.resid_[u].multiply(self.sim_[:, i].T).sum(axis=1)).ravel()
        den = np.asarray(self.binary_[u].multiply(np.abs(self.sim_[:, i]).T).sum(axis=1)).ravel()
        adj = np.divide(
            num, den + self.denom_reg, out=np.zeros_like(num), where=den + self.denom_reg > 1e-12
        )
        return self.bias_.predict(u, i) + adj

    def scores(self, users: np.ndarray) -> np.ndarray:
        return np.asarray(self.binary_[users] @ np.maximum(self.sim_, 0))
