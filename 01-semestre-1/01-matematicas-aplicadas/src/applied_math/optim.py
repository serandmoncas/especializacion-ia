"""Optimizadores de primer orden: descenso de gradiente (con momentum) y SGD por minilotes."""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass
class OptimResult:
    x: Array
    n_iter: int
    converged: bool
    history: list[float] = field(default_factory=list)  # valor de la pérdida por iteración/época


def gradient_descent(
    f: Callable[[Array], float],
    grad_f: Callable[[Array], Array],
    x0: Array,
    *,
    lr: float = 0.1,
    momentum: float = 0.0,
    max_iter: int = 10_000,
    tol: float = 1e-8,
) -> OptimResult:
    """Descenso de gradiente con momentum de Polyak (heavy ball).

    v ← β v − η ∇f(x);  x ← x + v.  Con β = 0 es GD clásico.
    Criterio de parada: ‖∇f(x)‖ < tol.
    """
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    history = [f(x)]
    for k in range(1, max_iter + 1):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            return OptimResult(x, k - 1, True, history)
        v = momentum * v - lr * g
        x = x + v
        fx = f(x)
        history.append(fx)
        if not np.isfinite(fx):
            return OptimResult(x, k, False, history)  # divergió: lr demasiado grande
    return OptimResult(x, max_iter, False, history)


def minibatch_sgd(
    loss: Callable[[Array, Array, Array], float],
    grad: Callable[[Array, Array, Array], Array],
    w0: Array,
    X: Array,
    y: Array,
    *,
    lr: float = 0.01,
    batch_size: int = 32,
    epochs: int = 50,
    rng: np.random.Generator | None = None,
) -> OptimResult:
    """SGD por minilotes. `loss(w, X, y)` y `grad(w, X, y)` operan sobre un lote.

    batch_size = n → GD completo; batch_size = 1 → SGD puro.
    El historial guarda la pérdida sobre todo el dataset al final de cada época.
    """
    rng = rng or np.random.default_rng(0)
    w = np.array(w0, dtype=float)
    n = X.shape[0]
    history = [loss(w, X, y)]
    for _ in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            b = idx[start : start + batch_size]
            w = w - lr * grad(w, X[b], y[b])
        history.append(loss(w, X, y))
        if not np.isfinite(history[-1]):
            return OptimResult(w, len(history) - 1, False, history)
    return OptimResult(w, epochs, True, history)
