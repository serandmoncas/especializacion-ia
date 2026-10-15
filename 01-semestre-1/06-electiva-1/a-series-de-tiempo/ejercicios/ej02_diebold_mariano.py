"""Ejercicio 2: prueba de Diebold–Mariano. Ver README.md."""

import numpy as np
from scipy import stats


def diebold_mariano(
    e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2
) -> tuple[float, float]:
    """(estadístico, p-valor bilateral). Negativo: el pronóstico 1 tiene menor pérdida."""
    raise NotImplementedError


def _ma_errors(rng, n, h):
    """Errores de pronóstico a h pasos: MA(h−1), como los de un pronóstico óptimo."""
    e = rng.normal(size=n + h)
    return np.convolve(e, np.ones(h), mode="valid")[:n]


def check() -> None:
    rng = np.random.default_rng(0)
    h, n, reps = 4, 300, 2000
    rej_dm = rej_t = 0
    for _ in range(reps):
        e1, e2 = _ma_errors(rng, n, h), _ma_errors(rng, n, h)
        rej_dm += diebold_mariano(e1, e2, h)[1] < 0.05
        d = e1**2 - e2**2
        rej_t += stats.ttest_1samp(d, 0).pvalue < 0.05
    assert abs(rej_dm / reps - 0.05) < 0.02, rej_dm / reps
    assert rej_t / reps > rej_dm / reps
    better, worse = rng.normal(size=500), rng.normal(size=500) * 1.3
    assert diebold_mariano(better, worse)[1] < 0.01
    print(
        f"ej02 ok — rechazo bajo H₀ (h = {h}): DM {rej_dm / reps:.3f},"
        f" t-test ingenuo {rej_t / reps:.3f}"
    )


if __name__ == "__main__":
    check()
