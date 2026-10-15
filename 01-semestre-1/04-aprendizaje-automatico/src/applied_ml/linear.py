"""Regresión logística binaria con regularización L2, entrenada con descenso de gradiente."""

import numpy as np
from applied_math import gradient_descent
from sklearn.base import BaseEstimator, ClassifierMixin


def _log_sigmoid(z: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0, -z)


class LogisticRegression(ClassifierMixin, BaseEstimator):
    """Minimiza (1/n) Σ logloss + (α/2)‖w‖² (el intercepto no se penaliza).

    Equivale a `sklearn.linear_model.LogisticRegression(C=1/(n·α))`.

    Dos solvers, y la elección importa:
    - "gd": descenso de gradiente con momentum. La tasa no es un hiperparámetro: la pérdida es
      L-suave con L = λ_max(XᵀX)/(4n) + α y se usa η = 1/L. Necesita ~√κ·log(1/ε) iteraciones:
      con features one-hot de categorías raras κ es enorme y no converge en tiempo razonable.
    - "newton" (IRLS): w ← w − H⁻¹∇L con H = XᵀSX/n + αI, S = diag(p(1−p)). Cada iteración
      resuelve un sistema d×d, pero converge cuadráticamente (~10 iteraciones) sin importar κ.
      Ideal cuando d es moderado (cientos); para d enorme, H no cabe y se vuelve a métodos de
      primer orden o cuasi-Newton (L-BFGS, el default de sklearn).
    """

    def __init__(
        self,
        alpha: float = 1e-3,
        solver: str = "newton",
        momentum: float = 0.9,
        max_iter: int | None = None,
        tol: float = 1e-7,
    ):
        self.alpha = alpha
        self.solver = solver
        self.momentum = momentum
        self.max_iter = max_iter
        self.tol = tol

    def _loss_grad(self, X: np.ndarray, y: np.ndarray):
        n = len(y)

        def loss(w: np.ndarray) -> float:
            z = X @ w[1:] + w[0]
            # −[y log σ(z) + (1−y) log σ(−z)], estable con logaddexp
            ll = -(y * _log_sigmoid(z) + (1 - y) * _log_sigmoid(-z)).mean()
            return float(ll + 0.5 * self.alpha * w[1:] @ w[1:])

        def grad(w: np.ndarray) -> np.ndarray:
            p = np.exp(_log_sigmoid(X @ w[1:] + w[0]))
            r = p - y
            g = np.empty_like(w)
            g[0] = r.mean()
            g[1:] = X.T @ r / n + self.alpha * w[1:]
            return g

        return loss, grad

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError("Solo clasificación binaria")
        y_enc = y_enc.astype(float)
        Xb = np.hstack([np.ones((len(X), 1)), X])
        loss, grad = self._loss_grad(X, y_enc)
        w0 = np.zeros(X.shape[1] + 1)
        if self.solver == "gd":
            L = np.linalg.eigvalsh(Xb.T @ Xb / len(X))[-1] / 4 + self.alpha
            res = gradient_descent(
                loss,
                grad,
                w0,
                lr=1 / L,
                momentum=self.momentum,
                max_iter=self.max_iter or 20_000,
                tol=self.tol,
            )
            w, self.n_iter_, self.converged_, self.loss_history_ = (
                res.x,
                res.n_iter,
                res.converged,
                res.history,
            )
        elif self.solver == "newton":
            w, self.n_iter_, self.converged_, self.loss_history_ = self._newton(
                Xb, y_enc, loss, grad, w0
            )
        else:
            raise ValueError(f"solver desconocido: {self.solver!r}")
        self.intercept_ = np.array([w[0]])
        self.coef_ = w[1:][None, :]
        return self

    def _newton(self, Xb, y, loss, grad, w):
        reg = np.full(Xb.shape[1], self.alpha)
        reg[0] = 0.0  # el intercepto no se regulariza
        history = [loss(w)]
        for k in range(1, (self.max_iter or 100) + 1):
            g = grad(w)
            if np.linalg.norm(g) < self.tol:
                return w, k - 1, True, history
            p = np.exp(_log_sigmoid(Xb @ w))
            H = (Xb.T * (p * (1 - p))) @ Xb / len(y) + np.diag(reg)
            step = np.linalg.solve(H, g)
            t = 1.0  # backtracking: el paso completo casi siempre se acepta, pero protege al inicio
            while loss(w - t * step) > history[-1] - 1e-4 * t * (g @ step) and t > 1e-8:
                t *= 0.5
            w = w - t * step
            history.append(loss(w))
        return w, self.max_iter or 100, False, history

    def decision_function(self, X) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, X) -> np.ndarray:
        p1 = np.exp(_log_sigmoid(self.decision_function(X)))
        return np.column_stack([1 - p1, p1])

    def predict(self, X) -> np.ndarray:
        return self.classes_[(self.decision_function(X) > 0).astype(int)]
