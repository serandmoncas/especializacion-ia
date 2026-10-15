"""Gradientes numéricos y verificación de gradientes analíticos."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def numerical_gradient(f: Callable[[Array], float], x: Array, *, eps: float = 1e-6) -> Array:
    """Gradiente por diferencias centrales: (f(x+εeᵢ) - f(x-εeᵢ)) / 2ε.

    Error de truncamiento O(ε²), frente a O(ε) de la diferencia hacia adelante.
    Cuesta 2·d evaluaciones de f: útil para verificar, no para entrenar.
    """
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        orig = x[i]
        x[i] = orig + eps
        f_plus = f(x)
        x[i] = orig - eps
        f_minus = f(x)
        x[i] = orig
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def gradient_check(
    f: Callable[[Array], float], grad_f: Callable[[Array], Array], x: Array, *, eps: float = 1e-6
) -> float:
    """Error relativo ‖g_a - g_n‖ / (‖g_a‖ + ‖g_n‖) entre gradiente analítico y numérico.

    Regla práctica: < 1e-7 correcto; > 1e-4 casi seguro hay un bug.
    """
    g_a = grad_f(np.array(x, dtype=float))
    g_n = numerical_gradient(f, np.array(x, dtype=float), eps=eps)
    denom = np.linalg.norm(g_a) + np.linalg.norm(g_n)
    if denom == 0:
        return 0.0
    return float(np.linalg.norm(g_a - g_n) / denom)
