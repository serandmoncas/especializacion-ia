"""Baselines: media global, sesgos de usuario e ítem, popularidad."""

import numpy as np
from scipy import sparse


class BiasModel:
    """r̂_ui = μ + b_u + b_i, con penalización L2 sobre los sesgos.

    Minimiza Σ (r_ui − μ − b_u − b_i)² + λ_u Σ b_u² + λ_i Σ b_i² alternando soluciones cerradas:
    b_i = Σ_u (r_ui − μ − b_u) / (λ_i + n_i), y análogo para b_u. La regularización encoge hacia 0
    los sesgos de ítems con pocas calificaciones: una película con un único 5 no es “la mejor”.
    """

    def __init__(self, reg_user: float = 10.0, reg_item: float = 10.0, n_iter: int = 15):
        self.reg_user, self.reg_item, self.n_iter = reg_user, reg_item, n_iter

    def fit(self, R: sparse.csr_matrix) -> "BiasModel":
        R = R.tocsr()
        coo = R.tocoo()
        self.mu = float(coo.data.mean())
        n_u, n_i = R.shape
        self.bu, self.bi = np.zeros(n_u), np.zeros(n_i)
        cnt_u = np.bincount(coo.row, minlength=n_u)
        cnt_i = np.bincount(coo.col, minlength=n_i)
        for _ in range(self.n_iter):
            res = coo.data - self.mu - self.bu[coo.row]
            self.bi = np.bincount(coo.col, weights=res, minlength=n_i) / (self.reg_item + cnt_i)
            res = coo.data - self.mu - self.bi[coo.col]
            self.bu = np.bincount(coo.row, weights=res, minlength=n_u) / (self.reg_user + cnt_u)
        return self

    def predict(self, u: np.ndarray, i: np.ndarray) -> np.ndarray:
        return self.mu + self.bu[u] + self.bi[i]


def popularity_scores(R: sparse.csr_matrix) -> np.ndarray:
    """Interacciones por ítem: el mismo ranking para todos. Baseline difícil de batir."""
    return np.asarray((R != 0).sum(axis=0)).ravel().astype(float)
