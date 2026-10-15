"""Solución ejercicio 2: entropía cruzada.

Con z = Xw: −[y log σ(z) + (1−y) log(1−σ(z))] = log(1 + eᶻ) − y·z = logaddexp(0, z) − y·z.
Esta forma nunca evalúa log(0) ni exp de algo grande.
Como dσ/dz = σ(1−σ), la derivada respecto a z es σ(z) − y, y por la regla de la cadena
∇L = (1/n) Xᵀ(p − y).
"""

import numpy as np
from applied_math import gradient_check


def sigmoid(z: np.ndarray) -> np.ndarray:
    # exp(-logaddexp(0, -z)) = 1/(1+e^{-z}) sin overflow para z muy negativo
    return np.exp(-np.logaddexp(0, -z))


def logistic_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    z = X @ w
    return float(np.mean(np.logaddexp(0, z) - y * z))


def logistic_gradient(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return X.T @ (sigmoid(X @ w) - y) / len(y)


def check() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X @ np.array([1.0, -1.0, 2.0]) > 0).astype(float)
    w = rng.normal(size=3)
    err = gradient_check(lambda v: logistic_loss(v, X, y), lambda v: logistic_gradient(v, X, y), w)
    assert err < 1e-7, f"gradiente incorrecto (error relativo {err:.1e})"
    assert np.isfinite(logistic_loss(1000 * w, X, y)), "pérdida no estable con |Xw| grande"
    print("ej02 ok")


if __name__ == "__main__":
    check()
