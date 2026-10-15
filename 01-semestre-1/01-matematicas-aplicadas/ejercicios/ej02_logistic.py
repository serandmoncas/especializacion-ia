"""Ejercicio 2: entropía cruzada. Ver README.md."""

import numpy as np
from applied_math import gradient_check


def logistic_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Debe ser estable: sin overflow ni log(0) aunque |Xw| sea ~1000."""
    raise NotImplementedError


def logistic_gradient(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    raise NotImplementedError


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
