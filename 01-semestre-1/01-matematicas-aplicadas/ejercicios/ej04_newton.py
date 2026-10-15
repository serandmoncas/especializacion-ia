"""Ejercicio 4: método de Newton en Rosenbrock. Ver README.md."""

import numpy as np


def rosenbrock(p: np.ndarray) -> float:
    x, y = p
    return float((1 - x) ** 2 + 100 * (y - x**2) ** 2)


def rosenbrock_grad(p: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def rosenbrock_hessian(p: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def newton(grad_f, hess_f, x0, *, max_iter=100, tol=1e-10) -> tuple[np.ndarray, int]:
    """x ← x − H⁻¹∇f. Resolver el sistema lineal, no invertir H."""
    raise NotImplementedError


def check() -> None:
    x, n_iter = newton(rosenbrock_grad, rosenbrock_hessian, np.array([-1.2, 1.0]))
    assert np.allclose(x, [1.0, 1.0], atol=1e-8), f"no llegó al mínimo: {x}"
    print(f"ej04 ok — Newton en {n_iter} iteraciones")


if __name__ == "__main__":
    check()
