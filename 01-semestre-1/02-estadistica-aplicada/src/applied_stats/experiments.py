"""Diseño y chequeos de experimentos: potencia, tamaño de muestra, SRM y CUPED."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats


def sample_size_two_proportions(
    p1: float, p2: float, *, alpha: float = 0.05, power: float = 0.8
) -> int:
    """n por grupo para detectar p1 → p2 con prueba z bilateral.

    n = [z_{1−α/2} √(2 p̄ q̄) + z_{1−β} √(p₁q₁ + p₂q₂)]² / (p₂ − p₁)²
    Nótese el 1/Δ²: detectar la mitad del efecto cuesta 4 veces más muestra.
    """
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return int(np.ceil(num**2 / (p2 - p1) ** 2))


def power_two_proportions(p1: float, p2: float, n: int, *, alpha: float = 0.05) -> float:
    """Potencia (aprox. normal) de la prueba z bilateral con n por grupo."""
    za = stats.norm.ppf(1 - alpha / 2)
    pbar = (p1 + p2) / 2
    se0 = np.sqrt(2 * pbar * (1 - pbar) / n)
    se1 = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
    d = abs(p2 - p1)
    return float(stats.norm.cdf((d - za * se0) / se1) + stats.norm.cdf((-d - za * se0) / se1))


def sample_ratio_mismatch(
    counts: ArrayLike, expected_ratios: ArrayLike | None = None
) -> tuple[float, float]:
    """Chequeo SRM: χ² de bondad de ajuste de los tamaños de grupo vs. la asignación diseñada.

    Un p muy bajo (convención: < 0.001) indica asignación o registro defectuosos, y entonces
    *ningún* resultado del experimento es confiable. Devuelve (χ², p-valor).
    """
    counts = np.asarray(counts, dtype=float)
    ratios = (
        np.full(len(counts), 1 / len(counts))
        if expected_ratios is None
        else np.asarray(expected_ratios, dtype=float) / np.sum(expected_ratios)
    )
    res = stats.chisquare(counts, counts.sum() * ratios)
    return float(res.statistic), float(res.pvalue)


def cuped(y: ArrayLike, x: ArrayLike, theta: float | None = None) -> NDArray[np.float64]:
    """CUPED: y_adj = y − θ (x − x̄), con x una covariable *previa* al experimento.

    θ* = cov(y, x) / var(x) minimiza var(y_adj) = var(y)(1 − ρ²). Como x no depende del
    tratamiento, E[y_adj] = E[y] en cada grupo: el efecto estimado no cambia, la varianza baja.
    Estimar θ con los datos combinados (no por grupo) para no sesgar la diferencia.
    """
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    if theta is None:
        theta = float(np.cov(y, x)[0, 1] / x.var(ddof=1))
    return y - theta * (x - x.mean())
