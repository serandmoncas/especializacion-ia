"""Correcciones por comparaciones múltiples. Devuelven p-valores *ajustados*.

Rechazar Hᵢ si p_ajustado_i ≤ α. FWER (Bonferroni, Holm): P(al menos un falso positivo) ≤ α.
FDR (Benjamini–Hochberg): E[falsos positivos / rechazos] ≤ α; más potente con muchas pruebas.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def bonferroni(pvalues: ArrayLike) -> NDArray[np.float64]:
    p = np.asarray(pvalues, dtype=float)
    return np.minimum(p * len(p), 1.0)


def holm(pvalues: ArrayLike) -> NDArray[np.float64]:
    """Holm (step-down): uniformemente más potente que Bonferroni, mismo control de FWER.

    Ordenados p₍₁₎ ≤ … ≤ p₍ₘ₎: p̃₍ᵢ₎ = max_{j≤i} min(1, (m − j + 1) p₍ⱼ₎).
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def benjamini_hochberg(pvalues: ArrayLike) -> NDArray[np.float64]:
    """BH (step-up): p̃₍ᵢ₎ = min_{j≥i} min(1, m p₍ⱼ₎ / j)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    scaled = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out
