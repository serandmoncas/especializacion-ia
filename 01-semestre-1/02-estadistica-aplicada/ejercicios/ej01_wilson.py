"""Ejercicio 1: intervalos para proporciones, Wald vs. Wilson. Ver README.md."""

import numpy as np
from scipy import stats


def wald_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    raise NotImplementedError


def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    raise NotImplementedError


def coverage(ci_fn, p: float, n: int, reps: int = 20_000, seed: int = 0) -> float:
    ks = np.random.default_rng(seed).binomial(n, p, size=reps)
    hits = [lo <= p <= hi for lo, hi in (ci_fn(int(k), n) for k in ks)]
    return float(np.mean(hits))


def check() -> None:
    lo, hi = wilson_ci(0, 50)
    assert lo == 0 or abs(lo) < 1e-12, "Wilson con k = 0 debe empezar en 0"
    assert hi > 0.05, "Wilson con k = 0 debe tener ancho positivo"
    ref = stats.binomtest(7, 50).proportion_ci(method="wilson")
    assert np.allclose(wilson_ci(7, 50), (ref.low, ref.high)), "no coincide con scipy"
    cov_wald, cov_wilson = coverage(wald_ci, 0.02, 50), coverage(wilson_ci, 0.02, 50)
    assert cov_wald < 0.7 and cov_wilson > 0.9, (cov_wald, cov_wilson)
    print(f"ej01 ok — cobertura con n=50, p=0.02: Wald {cov_wald:.3f}, Wilson {cov_wilson:.3f}")


if __name__ == "__main__":
    check()
