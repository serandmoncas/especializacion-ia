"""Ejercicio 3: búsqueda lineal con backtracking (condición de Armijo). Ver README.md."""

from collections.abc import Callable

import numpy as np
from applied_math import gradient_descent


def backtracking_step(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    g: np.ndarray,
    *,
    lr0: float = 1.0,
    rho: float = 0.5,
    c: float = 1e-4,
) -> float:
    """Devuelve un η que cumple f(x − ηg) ≤ f(x) − c·η·‖g‖².

    Empezar en lr0 y multiplicar por rho mientras no se cumpla la condición.
    Decidir qué hacer si η se vuelve absurdamente pequeño (¿límite de iteraciones?, ¿error?).
    """
    # TODO: implementar (5-10 líneas)
    raise NotImplementedError


def gd_backtracking(f, grad_f, x0, *, max_iter=10_000, tol=1e-8) -> tuple[np.ndarray, int]:
    x = np.array(x0, dtype=float)
    for k in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            return x, k
        x = x - backtracking_step(f, x, g) * g
    return x, max_iter


def check() -> None:
    H = np.diag([1.0, 10.0])

    def f(x):
        return 0.5 * x @ H @ x

    def grad_f(x):
        return H @ x

    x0 = np.array([4.0, 2.0])
    x, n_iter = gd_backtracking(f, grad_f, x0)
    assert np.allclose(x, 0, atol=1e-7), f"no convergió: {x}"
    fixed = gradient_descent(f, grad_f, x0, lr=0.1).n_iter
    print(f"ej03 ok — backtracking: {n_iter} iteraciones; η fijo = 0.1: {fixed}")


if __name__ == "__main__":
    check()
