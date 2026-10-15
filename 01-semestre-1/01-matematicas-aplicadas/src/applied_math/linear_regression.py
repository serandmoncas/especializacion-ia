"""Regresión lineal desde cero: forma cerrada y descenso de gradiente."""

from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray

from applied_math.optim import OptimResult, gradient_descent, minibatch_sgd

Array = NDArray[np.float64]


def add_bias(X: Array) -> Array:
    """Antepone una columna de unos para absorber el intercepto en w."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def mse(w: Array, X: Array, y: Array) -> float:
    """L(w) = (1/n) ‖Xw − y‖²."""
    r = X @ w - y
    return float(r @ r / X.shape[0])


def mse_gradient(w: Array, X: Array, y: Array) -> Array:
    """∇L(w) = (2/n) Xᵀ(Xw − y)."""
    return 2.0 * X.T @ (X @ w - y) / X.shape[0]


def fit_normal_equation(X: Array, y: Array) -> Array:
    """Mínimos cuadrados en forma cerrada (con sesgo). Resuelve XᵀXw = Xᵀy vía lstsq."""
    w, *_ = np.linalg.lstsq(add_bias(X), y, rcond=None)
    return w


class LinearRegressionGD:
    """Regresión lineal entrenada con GD completo o SGD por minilotes (API estilo sklearn)."""

    def __init__(
        self,
        *,
        solver: Literal["gd", "sgd"] = "gd",
        lr: float = 0.1,
        momentum: float = 0.0,
        max_iter: int = 5_000,
        batch_size: int = 32,
        epochs: int = 50,
        seed: int = 0,
    ) -> None:
        self.solver = solver
        self.lr = lr
        self.momentum = momentum
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed

    def fit(self, X: Array, y: Array) -> Self:
        Xb = add_bias(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float)
        w0 = np.zeros(Xb.shape[1])
        if self.solver == "gd":
            result = gradient_descent(
                lambda w: mse(w, Xb, y),
                lambda w: mse_gradient(w, Xb, y),
                w0,
                lr=self.lr,
                momentum=self.momentum,
                max_iter=self.max_iter,
            )
        elif self.solver == "sgd":
            result = minibatch_sgd(
                mse,
                mse_gradient,
                w0,
                Xb,
                y,
                lr=self.lr,
                batch_size=self.batch_size,
                epochs=self.epochs,
                rng=np.random.default_rng(self.seed),
            )
        else:
            raise ValueError(f"solver desconocido: {self.solver!r}")
        self.result_: OptimResult = result
        self.intercept_ = float(result.x[0])
        self.coef_ = result.x[1:]
        return self

    def predict(self, X: Array) -> Array:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_
