"""Solución ejercicio 2: bootstrap BCa.

z₀ corrige el sesgo mediano (si θ̂* cae por debajo de θ̂ más de la mitad de las veces,
la distribución está desplazada). a corrige que el error estándar varíe con θ; se estima con
la asimetría de los valores jackknife. Con z₀ = a = 0, BCa se reduce al percentil.
"""

from collections.abc import Callable

import numpy as np
from scipy import stats

Statistic = Callable[..., np.ndarray]  # stat(x, axis=-1), como np.mean


def bca_ci(
    x: np.ndarray,
    stat: Statistic,
    *,
    n_boot: int = 10_000,
    conf: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    n = len(x)
    theta = stat(x, axis=-1)
    boots = stat(x[rng.integers(0, n, size=(n_boot, n))], axis=-1)

    z0 = stats.norm.ppf(np.mean(boots < theta))
    loo = np.broadcast_to(x, (n, n))[~np.eye(n, dtype=bool)].reshape(n, n - 1)  # dejar uno fuera
    jack = stat(loo, axis=-1)
    d = jack.mean() - jack
    a = np.sum(d**3) / (6 * np.sum(d**2) ** 1.5)

    alphas = np.array([(1 - conf) / 2, (1 + conf) / 2])
    z = stats.norm.ppf(alphas)
    adjusted = stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo, hi = np.quantile(boots, adjusted)
    return float(lo), float(hi)


def check() -> None:
    x = np.random.default_rng(1).exponential(3.0, size=60)
    lo, hi = bca_ci(x, np.mean, n_boot=40_000, rng=np.random.default_rng(2))
    ref = stats.bootstrap(
        (x,), np.mean, method="BCa", n_resamples=40_000, rng=np.random.default_rng(3)
    ).confidence_interval
    assert np.isclose(lo, ref.low, rtol=0.02) and np.isclose(hi, ref.high, rtol=0.02), (
        (lo, hi),
        (ref.low, ref.high),
    )
    print(f"ej02 ok — BCa ({lo:.3f}, {hi:.3f}) vs scipy ({ref.low:.3f}, {ref.high:.3f})")


if __name__ == "__main__":
    check()
