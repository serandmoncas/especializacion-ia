"""Solución ejercicio 1: Ridge.

∇L = (2/n) Xᵀ(Xw − y) + 2λw. Igualando a cero: (XᵀX + nλI) w = Xᵀy.
XᵀX es PSD, así que XᵀX + nλI tiene autovalores ≥ nλ > 0: siempre invertible.
Interpretación: λ desplaza todo el espectro, y el número de condición pasa de
λmax/λmin a (λmax + nλ)/(λmin + nλ).
"""

import numpy as np
from applied_math import gradient_check


def ridge_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray, lam: float) -> float:
    r = X @ w - y
    return float(r @ r / len(y) + lam * w @ w)


def ridge_gradient(w: np.ndarray, X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    return 2 * X.T @ (X @ w - y) / len(y) + 2 * lam * w


def ridge_closed_form(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    n, d = X.shape
    return np.linalg.solve(X.T @ X + n * lam * np.eye(d), X.T @ y)


def check() -> None:
    rng = np.random.default_rng(0)
    X, y, lam = rng.normal(size=(100, 5)), rng.normal(size=100), 0.3
    w = rng.normal(size=5)
    err = gradient_check(
        lambda v: ridge_loss(v, X, y, lam), lambda v: ridge_gradient(v, X, y, lam), w
    )
    assert err < 1e-7, f"gradiente incorrecto (error relativo {err:.1e})"
    w_star = ridge_closed_form(X, y, lam)
    assert np.allclose(ridge_gradient(w_star, X, y, lam), 0, atol=1e-10), "w* no anula el gradiente"
    print("ej01 ok")


if __name__ == "__main__":
    check()
