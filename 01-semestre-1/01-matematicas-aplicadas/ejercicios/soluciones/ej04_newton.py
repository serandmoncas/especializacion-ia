"""Solución ejercicio 4: Newton en Rosenbrock.

∇f = [−2(1−x) − 400x(y−x²),  200(y−x²)]
H  = [[2 − 400(y − 3x²), −400x], [−400x, 200]]

Invarianza: si x = Az, el paso de Newton en z es A⁻¹ por el paso en x; las iteraciones
coinciden. El paso de GD en z es Aᵀ∇f, que en general no corresponde a ningún paso de GD en x.
Newton converge cuadráticamente cerca del mínimo; GD necesita miles de iteraciones en
el valle curvo de Rosenbrock (κ ≈ 2500 en (1, 1)).
"""

import numpy as np


def rosenbrock(p: np.ndarray) -> float:
    x, y = p
    return float((1 - x) ** 2 + 100 * (y - x**2) ** 2)


def rosenbrock_grad(p: np.ndarray) -> np.ndarray:
    x, y = p
    return np.array([-2 * (1 - x) - 400 * x * (y - x**2), 200 * (y - x**2)])


def rosenbrock_hessian(p: np.ndarray) -> np.ndarray:
    x, y = p
    return np.array([[2 - 400 * (y - 3 * x**2), -400 * x], [-400 * x, 200.0]])


def newton(grad_f, hess_f, x0, *, max_iter=100, tol=1e-10) -> tuple[np.ndarray, int]:
    """x ← x − H⁻¹∇f. Resolver el sistema lineal, no invertir H."""
    x = np.array(x0, dtype=float)
    for k in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            return x, k
        x = x - np.linalg.solve(hess_f(x), g)
    return x, max_iter


def check() -> None:
    x, n_iter = newton(rosenbrock_grad, rosenbrock_hessian, np.array([-1.2, 1.0]))
    assert np.allclose(x, [1.0, 1.0], atol=1e-8), f"no llegó al mínimo: {x}"
    print(f"ej04 ok — Newton en {n_iter} iteraciones")


if __name__ == "__main__":
    check()
