"""Descomposición clásica aditiva: y = tendencia + estacionalidad + residuo."""

import numpy as np


def centered_moving_average(y: np.ndarray, m: int) -> np.ndarray:
    """Media móvil centrada de orden m. Para m par se usa 2×m (promedio de dos medias de m),
    así cada estación pesa lo mismo y el resultado queda alineado con la observación central.
    Los extremos sin ventana completa quedan en NaN."""
    y = np.asarray(y, dtype=float)
    w = np.ones(m) / m if m % 2 else np.r_[0.5, np.ones(m - 1), 0.5] / m
    trend = np.full(len(y), np.nan)
    half = len(w) // 2
    trend[half : len(y) - half] = np.convolve(y, w, mode="valid")
    return trend


def classical_decompose(y: np.ndarray, period: int) -> dict[str, np.ndarray]:
    """Tendencia = media móvil centrada; estacionalidad = promedio por posición del ciclo del
    residuo sin tendencia, centrado para sumar cero; residuo = lo que queda."""
    y = np.asarray(y, dtype=float)
    trend = centered_moving_average(y, period)
    detrended = y - trend
    pos = np.arange(len(y)) % period
    means = np.array([np.nanmean(detrended[pos == k]) for k in range(period)])
    means -= means.mean()
    seasonal = means[pos]
    return {"trend": trend, "seasonal": seasonal, "resid": y - trend - seasonal}
