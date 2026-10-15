"""Remuestreo: bootstrap y pruebas de permutación, vectorizados por bloques."""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from applied_stats.inference import Alternative

# Estadístico vectorizado: recibe una matriz [n_rep, n] y devuelve [n_rep].
Statistic = Callable[[NDArray[np.float64]], NDArray[np.float64]]
_CHUNK_CELLS = 5_000_000  # tope de celdas por bloque (~40 MB en float64)


def _mean(m: NDArray[np.float64]) -> NDArray[np.float64]:
    return m.mean(axis=1)


def _resample_stats(x: NDArray, stat: Statistic, n_boot: int, rng: np.random.Generator) -> NDArray:
    """Aplica `stat` a `n_boot` remuestras con reemplazo, en bloques para acotar memoria."""
    n = len(x)
    rows = max(1, _CHUNK_CELLS // n)
    out = np.empty(n_boot)
    for start in range(0, n_boot, rows):
        k = min(rows, n_boot - start)
        out[start : start + k] = stat(x[rng.integers(0, n, size=(k, n))])
    return out


def bootstrap_ci(
    x: ArrayLike,
    stat: Statistic = _mean,
    *,
    n_boot: int = 10_000,
    conf: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, NDArray[np.float64]]:
    """Intervalo bootstrap percentil. Devuelve (bajo, alto, distribución bootstrap).

    Idea: la distribución empírica F̂ sustituye a la poblacional F; la variabilidad de
    θ̂* bajo F̂ aproxima la de θ̂ bajo F. Falla con estadísticos no suaves (máximo)
    y con muestras muy pequeñas.
    """
    rng = rng or np.random.default_rng(0)
    boots = _resample_stats(np.asarray(x, dtype=float), stat, n_boot, rng)
    alpha = 1 - conf
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), boots


def bootstrap_diff_ci(
    a: ArrayLike,
    b: ArrayLike,
    stat: Statistic = _mean,
    *,
    n_boot: int = 10_000,
    conf: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, NDArray[np.float64]]:
    """Intervalo percentil para stat(b) − stat(a), remuestreando cada grupo por separado."""
    rng = rng or np.random.default_rng(0)
    ba = _resample_stats(np.asarray(a, dtype=float), stat, n_boot, rng)
    bb = _resample_stats(np.asarray(b, dtype=float), stat, n_boot, rng)
    diffs = bb - ba
    alpha = 1 - conf
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), diffs


def permutation_test(
    a: ArrayLike,
    b: ArrayLike,
    stat: Statistic = _mean,
    *,
    n_perm: int = 10_000,
    alternative: Alternative = "two-sided",
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Prueba de permutación para stat(b) − stat(a). Devuelve (observado, p-valor).

    Bajo H₀ (misma distribución) las etiquetas son intercambiables: barajarlas genera
    la distribución nula exacta del estadístico, sin supuestos paramétricos.
    p = (1 + #{|T*| ≥ |T|}) / (1 + B): el +1 cuenta la permutación observada y evita p = 0.
    """
    rng = rng or np.random.default_rng(0)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pooled = np.concatenate([a, b])
    na = len(a)
    observed = float(stat(b[None, :])[0] - stat(a[None, :])[0])
    rows = max(1, _CHUNK_CELLS // len(pooled))
    null = np.empty(n_perm)
    for start in range(0, n_perm, rows):
        k = min(rows, n_perm - start)
        perm = rng.permuted(np.broadcast_to(pooled, (k, len(pooled))), axis=1)
        null[start : start + k] = stat(perm[:, na:]) - stat(perm[:, :na])
    if alternative == "two-sided":
        extreme = np.abs(null) >= abs(observed)
    elif alternative == "greater":
        extreme = null >= observed
    else:
        extreme = null <= observed
    return observed, float((1 + extreme.sum()) / (1 + n_perm))
