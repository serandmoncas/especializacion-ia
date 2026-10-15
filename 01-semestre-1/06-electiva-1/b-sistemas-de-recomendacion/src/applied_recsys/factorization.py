"""Factorización de matrices: explícita (SGD y ALS), implícita (ALS ponderado) y PureSVD."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds


class FunkSVD:
    """r̂_ui = μ + b_u + b_i + p_uᵀ q_i, con SGD por minilotes (Funk, 2006; Koren et al., 2009).

    Cada minilote calcula los gradientes de sus calificaciones y los acumula con `np.add.at`, que
    suma correctamente cuando un usuario o ítem aparece varias veces en el mismo lote.
    """

    def __init__(
        self,
        k: int = 32,
        lr: float = 0.01,
        reg: float = 0.05,
        epochs: int = 20,
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

    def fit(self, R: sparse.csr_matrix, R_val: sparse.csr_matrix | None = None) -> "FunkSVD":
        rng = np.random.default_rng(self.seed)
        coo = R.tocoo()
        u, i, r = coo.row, coo.col, coo.data
        n_u, n_i = R.shape
        self.mu = float(r.mean())
        self.bu, self.bi = np.zeros(n_u), np.zeros(n_i)
        self.P = rng.normal(0, 0.1, (n_u, self.k))
        self.Q = rng.normal(0, 0.1, (n_i, self.k))
        self.history = []
        for _ in range(self.epochs):
            for b in np.array_split(rng.permutation(len(r)), max(1, len(r) // self.batch)):
                ub, ib = u[b], i[b]
                err = r[b] - self.predict(ub, ib)
                pu, qi = self.P[ub], self.Q[ib]
                np.add.at(self.bu, ub, self.lr * (err - self.reg * self.bu[ub]))
                np.add.at(self.bi, ib, self.lr * (err - self.reg * self.bi[ib]))
                np.add.at(self.P, ub, self.lr * (err[:, None] * qi - self.reg * pu))
                np.add.at(self.Q, ib, self.lr * (err[:, None] * pu - self.reg * qi))
            if R_val is not None:
                v = R_val.tocoo()
                self.history.append(
                    float(np.sqrt(np.mean((v.data - self.predict(v.row, v.col)) ** 2)))
                )
        return self

    def predict(self, u: np.ndarray, i: np.ndarray) -> np.ndarray:
        return self.mu + self.bu[u] + self.bi[i] + np.einsum("nk,nk->n", self.P[u], self.Q[i])


def ridge_rows(target: sparse.csr_matrix, fixed: np.ndarray, reg: float) -> np.ndarray:
    """Para cada fila de `target`, resuelve min_x Σ_{j observados} (t_j − f_jᵀ x)² + reg·n_obs·‖x‖².

    Es un paso de ALS: con un factor fijo, el otro tiene solución cerrada. La regularización
    proporcional al número de observaciones es la de Zhou et al. (2008, premio Netflix).
    """
    target = target.tocsr()
    k = fixed.shape[1]
    out = np.zeros((target.shape[0], k))
    eye = np.eye(k)
    for row in range(target.shape[0]):
        start, end = target.indptr[row], target.indptr[row + 1]
        if start == end:
            continue
        cols, vals = target.indices[start:end], target.data[start:end]
        F = fixed[cols]
        out[row] = np.linalg.solve(F.T @ F + reg * (end - start) * eye, F.T @ vals)
    return out


class ExplicitALS:
    """Mínimos cuadrados alternados sobre los residuos de un modelo de sesgos."""

    def __init__(self, k: int = 32, reg: float = 0.05, n_iter: int = 10, seed: int = 0):
        self.k, self.reg, self.n_iter, self.seed = k, reg, n_iter, seed

    def fit(self, R: sparse.csr_matrix, bias_model) -> "ExplicitALS":
        coo = R.tocoo()
        resid = sparse.csr_matrix(
            (coo.data - bias_model.predict(coo.row, coo.col), (coo.row, coo.col)), shape=R.shape
        )
        rng = np.random.default_rng(self.seed)
        self.bias_ = bias_model
        self.Q = rng.normal(0, 0.1, (R.shape[1], self.k))
        self.loss_history = []
        resid_T = resid.T.tocsr()
        for _ in range(self.n_iter):
            self.P = ridge_rows(resid, self.Q, self.reg)
            self.Q = ridge_rows(resid_T, self.P, self.reg)
            pred = np.einsum("nk,nk->n", self.P[coo.row], self.Q[coo.col])
            self.loss_history.append(float(np.mean((resid.tocoo().data - pred) ** 2)))
        return self

    def predict(self, u: np.ndarray, i: np.ndarray) -> np.ndarray:
        return self.bias_.predict(u, i) + np.einsum("nk,nk->n", self.P[u], self.Q[i])


class ImplicitALS:
    """ALS para retroalimentación implícita (Hu, Koren & Volinsky, 2008).

    Toda celda usuario–ítem es una observación: preferencia p_ui = 1 si hubo interacción, 0 si no,
    con confianza c_ui = 1 + α·r_ui. Minimiza Σ_{todas} c_ui (p_ui − x_uᵀ y_i)² + λ(‖X‖² + ‖Y‖²).

    El truco que lo hace viable: YᵀC_uY = YᵀY + Yᵀ(C_u − I)Y, y C_u − I es cero fuera de los ítems
    del usuario. YᵀY se calcula una vez por iteración; cada usuario solo paga por sus interacciones.
    """

    def __init__(
        self, k: int = 64, reg: float = 0.1, alpha: float = 10.0, n_iter: int = 10, seed: int = 0
    ):
        self.k, self.reg, self.alpha, self.n_iter, self.seed = k, reg, alpha, n_iter, seed

    @staticmethod
    def _solve(C: sparse.csr_matrix, Y: np.ndarray, reg: float) -> np.ndarray:
        YtY = Y.T @ Y
        k = Y.shape[1]
        out = np.zeros((C.shape[0], k))
        for row in range(C.shape[0]):
            start, end = C.indptr[row], C.indptr[row + 1]
            if start == end:
                continue
            cols, conf = C.indices[start:end], C.data[start:end]  # conf = c_ui ≥ 1
            Yn = Y[cols]
            A = YtY + (Yn.T * (conf - 1)) @ Yn + reg * np.eye(k)
            b = Yn.T @ conf  # Yᵀ C_u p_u, con p_u = 1 en los observados
            out[row] = np.linalg.solve(A, b)
        return out

    def fit(self, R: sparse.csr_matrix) -> "ImplicitALS":
        C = R.tocsr().astype(np.float64)
        C.data = 1.0 + self.alpha * C.data
        rng = np.random.default_rng(self.seed)
        self.X = rng.normal(0, 0.01, (R.shape[0], self.k))
        self.Y = rng.normal(0, 0.01, (R.shape[1], self.k))
        CT = C.T.tocsr()
        for _ in range(self.n_iter):
            self.X = self._solve(C, self.Y, self.reg)
            self.Y = self._solve(CT, self.X, self.reg)
        return self

    def scores(self, users: np.ndarray) -> np.ndarray:
        return self.X[users] @ self.Y.T


class PureSVD:
    """Top-N por SVD truncada de la matriz completa con ceros (Cremonesi, Koren & Turrin, 2010).

    Los faltantes cuentan como 0: “no visto” se trata como señal. Pésimo para predecir ratings,
    sorprendentemente bueno para rankear. Puntaje: r_u V Vᵀ (proyección sobre k factores).
    """

    def __init__(self, k: int = 50):
        self.k = k

    def fit(self, R: sparse.csr_matrix) -> "PureSVD":
        _, _, vt = svds(R.astype(np.float64), k=self.k)
        self.V = vt.T
        self.R = R.tocsr().astype(np.float64)
        return self

    def scores(self, users: np.ndarray) -> np.ndarray:
        return np.asarray(self.R[users] @ self.V) @ self.V.T
