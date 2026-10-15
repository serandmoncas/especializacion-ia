"""Ejercicio 3: frontera de Pocock por simulación. Ver README.md."""

import numpy as np
from scipy import stats


def max_abs_z(n_looks: int, reps: int = 200_000, seed: int = 0) -> np.ndarray:
    """Simula max_k |Z_k| bajo H₀ para `reps` experimentos."""
    increments = np.random.default_rng(seed).normal(size=(reps, n_looks))
    z = np.cumsum(increments, axis=1) / np.sqrt(np.arange(1, n_looks + 1))
    return np.abs(z).max(axis=1)


def pocock_boundary(n_looks: int, alpha: float = 0.05, tol: float = 1e-4) -> float:
    raise NotImplementedError


def check() -> None:
    c = pocock_boundary(5)
    assert abs(c - 2.413) < 0.02, f"c = {c:.3f}, tablas: 2.413"
    assert abs(pocock_boundary(1) - 1.96) < 0.02
    per_look = 2 * stats.norm.sf(c)
    print(f"ej03 ok — K=5: c = {c:.3f}, α por mirada = {per_look:.4f}")


if __name__ == "__main__":
    check()
