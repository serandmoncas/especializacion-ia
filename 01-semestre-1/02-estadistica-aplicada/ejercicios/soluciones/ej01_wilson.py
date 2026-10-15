"""Solución ejercicio 1: Wald vs. Wilson.

Wilson: elevar al cuadrado |p̂ − p| ≤ z√(p(1−p)/n) da una cuadrática en p cuyas raíces son
    centro = (p̂ + z²/2n) / (1 + z²/n)
    radio  = z/(1 + z²/n) · √(p̂(1−p̂)/n + z²/4n²)
El centro se encoge hacia 1/2 (como agregar z²/2 éxitos y fracasos ficticios) y el radio nunca
es cero: por eso funciona con p̂ = 0, donde Wald colapsa.
"""

import numpy as np
from scipy import stats


def wald_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    z = stats.norm.ppf((1 + conf) / 2)
    p = k / n
    half = z * np.sqrt(p * (1 - p) / n)
    return p - half, p + half


def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    z = stats.norm.ppf((1 + conf) / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z / denom * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return center - half, center + half


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
