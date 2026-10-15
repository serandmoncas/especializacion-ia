"""Inferencia frecuentista clásica: intervalos t, Welch y z para proporciones.

Convención: las diferencias son siempre `b − a` (tratamiento menos control).
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

Alternative = Literal["two-sided", "greater", "less"]


@dataclass(frozen=True)
class TestResult:
    diff: float  # estimación puntual de b − a
    ci: tuple[float, float]  # intervalo de confianza de la diferencia
    statistic: float
    pvalue: float


def _pvalue(stat: float, dist, alternative: Alternative) -> float:
    if alternative == "two-sided":
        return float(2 * dist.sf(abs(stat)))
    if alternative == "greater":
        return float(dist.sf(stat))
    return float(dist.cdf(stat))


def mean_ci(x: ArrayLike, conf: float = 0.95) -> tuple[float, float]:
    """Intervalo t para la media: x̄ ± t_{n−1} · s/√n."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    se = x.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf((1 + conf) / 2, df=n - 1)
    return float(x.mean() - t * se), float(x.mean() + t * se)


def welch_ttest(
    a: ArrayLike, b: ArrayLike, *, conf: float = 0.95, alternative: Alternative = "two-sided"
) -> TestResult:
    """Prueba t de Welch (varianzas distintas) para la diferencia de medias b − a.

    Grados de libertad de Welch–Satterthwaite:
        ν = (s²ₐ/nₐ + s²ᵦ/nᵦ)² / [ (s²ₐ/nₐ)²/(nₐ−1) + (s²ᵦ/nᵦ)²/(nᵦ−1) ]
    Preferirla por defecto sobre la t de Student: cuesta casi nada de potencia si las
    varianzas son iguales y protege si no lo son.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    diff = b.mean() - a.mean()
    t = diff / se
    dist = stats.t(df)
    half = dist.ppf((1 + conf) / 2) * se
    return TestResult(
        float(diff),
        (float(diff - half), float(diff + half)),
        float(t),
        _pvalue(t, dist, alternative),
    )


def two_proportion_ztest(
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int,
    *,
    conf: float = 0.95,
    alternative: Alternative = "two-sided",
) -> TestResult:
    """Prueba z para pₐ vs pᵦ.

    El estadístico usa el error estándar *agrupado* (bajo H₀ pₐ = pᵦ = p̂), pero el
    intervalo usa el *no agrupado*, porque el intervalo no asume H₀.
    Es equivalente a χ² de Pearson 2×2 sin corrección: z² = χ².
    """
    pa, pb = successes_a / n_a, successes_b / n_b
    pooled = (successes_a + successes_b) / (n_a + n_b)
    se0 = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (pb - pa) / se0
    se = np.sqrt(pa * (1 - pa) / n_a + pb * (1 - pb) / n_b)
    half = stats.norm.ppf((1 + conf) / 2) * se
    diff = pb - pa
    return TestResult(
        float(diff),
        (float(diff - half), float(diff + half)),
        float(z),
        _pvalue(z, stats.norm, alternative),
    )
