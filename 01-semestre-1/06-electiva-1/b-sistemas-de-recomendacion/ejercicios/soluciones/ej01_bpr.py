"""Solución ejercicio 1: BPR-MF.

Para cada tripleta (u, i, j): x̂_uij = p_uᵀ(q_i − q_j) + (b_i − b_j). El gradiente de
log σ(x̂_uij) es (1 − σ(x̂_uij)) por la derivada de x̂_uij: empuja a i hacia arriba y a j hacia
abajo, con más fuerza cuanto peor ordenado esté el par. Los negativos j se muestrean al azar entre
los no vistos (casi todo el catálogo): barato y, en la práctica, suficiente.

Dos lecciones del check: (1) sin regularización suficiente los factores memorizan qué pares se
observaron y el AUC cae por debajo del de la popularidad; (2) personalizar solo rinde si hay gusto
personal que aprender: con poca señal (signal = 1) BPR apenas empata a la popularidad.
"""

import numpy as np
from scipy import sparse


class BPR:
    def __init__(
        self,
        k: int = 32,
        lr: float = 0.05,
        reg: float = 0.01,
        epochs: int = 30,
        batch: int = 1024,
        seed: int = 0,
    ):
        self.k, self.lr, self.reg, self.epochs, self.batch, self.seed = (
            k,
            lr,
            reg,
            epochs,
            batch,
            seed,
        )

    def fit(self, R: sparse.csr_matrix) -> "BPR":
        rng = np.random.default_rng(self.seed)
        coo = R.tocoo()
        users, pos = coo.row, coo.col
        n_u, n_i = R.shape
        seen = R.tocsr().astype(bool)
        self.P = rng.normal(0, 0.1, (n_u, self.k))
        self.Q = rng.normal(0, 0.1, (n_i, self.k))
        self.b = np.zeros(n_i)
        for _ in range(self.epochs):
            for idx in np.array_split(
                rng.permutation(len(users)), max(1, len(users) // self.batch)
            ):
                u, i = users[idx], pos[idx]
                j = rng.integers(0, n_i, len(idx))
                ok = ~np.asarray(
                    seen[u, j]
                ).ravel()  # descartar negativos que en realidad son positivos
                u, i, j = u[ok], i[ok], j[ok]
                pu, qi, qj = self.P[u], self.Q[i], self.Q[j]
                x = np.einsum("nk,nk->n", pu, qi - qj) + self.b[i] - self.b[j]
                g = 1.0 / (1.0 + np.exp(x))  # = 1 − σ(x)
                np.add.at(self.P, u, self.lr * (g[:, None] * (qi - qj) - self.reg * pu))
                np.add.at(self.Q, i, self.lr * (g[:, None] * pu - self.reg * qi))
                np.add.at(self.Q, j, self.lr * (-g[:, None] * pu - self.reg * qj))
                np.add.at(self.b, i, self.lr * (g - self.reg * self.b[i]))
                np.add.at(self.b, j, self.lr * (-g - self.reg * self.b[j]))
        return self

    def scores(self, users: np.ndarray) -> np.ndarray:
        return self.P[users] @ self.Q.T + self.b


def ranking_auc(
    scores: np.ndarray,
    R_train: sparse.csr_matrix,
    test_pos: dict[int, set[int]],
    rng: np.random.Generator,
    n_neg: int = 100,
) -> float:
    """P(puntaje de un positivo de test > el de un no visto al azar), promedio por usuario."""
    aucs = []
    seen = R_train.tocsr()
    for row, u in enumerate(sorted(test_pos)):
        pos = np.array(sorted(test_pos[u]))
        cand = np.setdiff1d(np.arange(scores.shape[1]), np.r_[seen[u].indices, pos])
        neg = rng.choice(cand, min(n_neg, len(cand)), replace=False)
        aucs.append(np.mean(scores[row, pos][:, None] > scores[row, neg][None, :]))
    return float(np.mean(aucs))


def check() -> None:
    from applied_recsys.baselines import popularity_scores
    from applied_recsys.data import random_split, to_csr
    from applied_recsys.factorization import FunkSVD
    from applied_recsys.synthetic import synthetic_ratings

    # Con poco gusto personal (signal = 1) BPR solo empata a la popularidad: no hay nada que
    # personalizar. Con signal = 4 el gusto pesa y el ranking personalizado gana con claridad.
    df, _ = synthetic_ratings(1000, 600, density=0.05, popularity_skew=0.5, signal=4.0, seed=0)
    shape = (1000, 600)
    tr, te = random_split(df, 0.2, seed=0)
    R = to_csr(tr, shape)
    test_pos = {int(u): set(g.i) for u, g in te[te.rating >= 4].groupby("u")}
    users = np.array(sorted(test_pos))
    rng = np.random.default_rng(1)
    bpr = BPR(k=32, reg=0.05, epochs=30).fit(R)
    funk = FunkSVD(k=16, epochs=30).fit(R)
    pop = popularity_scores(R)
    auc = {
        "BPR": ranking_auc(bpr.scores(users), R, test_pos, rng),
        "popularidad": ranking_auc(np.tile(pop, (len(users), 1)), R, test_pos, rng),
        "FunkSVD": ranking_auc(
            funk.mu + funk.bu[users, None] + funk.bi[None, :] + funk.P[users] @ funk.Q.T,
            R,
            test_pos,
            rng,
        ),
    }
    assert auc["BPR"] > auc["popularidad"] + 0.05 and auc["BPR"] > auc["FunkSVD"], auc
    print("ej01 ok — AUC de ranking:", {k: round(v, 3) for k, v in auc.items()})


if __name__ == "__main__":
    check()
