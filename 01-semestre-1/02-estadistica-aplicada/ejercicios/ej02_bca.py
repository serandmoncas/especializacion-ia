"""Ejercicio 2: bootstrap BCa desde cero. Ver README.md."""

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
    raise NotImplementedError


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
