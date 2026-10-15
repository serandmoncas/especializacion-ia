"""Autocorrelación (ACF), autocorrelación parcial (PACF) y diferenciación."""

import numpy as np


def acf(y: np.ndarray, nlags: int) -> np.ndarray:
    """ρ(k) = γ(k)/γ(0) con γ(k) = (1/n) Σ (y_t − ȳ)(y_{t+k} − ȳ).

    Dividir por n (no por n − k) hace la estimación sesgada pero garantiza una matriz de
    autocovarianzas definida positiva, que es lo que necesita Durbin–Levinson.
    """
    y = np.asarray(y, dtype=float) - np.mean(y)
    n = len(y)
    gamma0 = y @ y / n
    return np.array([1.0] + [(y[:-k] @ y[k:]) / n / gamma0 for k in range(1, nlags + 1)])


def pacf(y: np.ndarray, nlags: int) -> np.ndarray:
    """PACF por la recursión de Durbin–Levinson: φ_kk es el último coeficiente del AR(k) óptimo.

    Mide la correlación entre y_t e y_{t−k} descontando los rezagos intermedios: en un AR(p)
    se anula para k > p, lo que sirve para identificar p.
    """
    r = acf(y, nlags)
    out = np.zeros(nlags + 1)
    out[0] = 1.0
    phi = np.zeros(nlags + 1)
    for k in range(1, nlags + 1):
        num = r[k] - np.dot(phi[1:k], r[k - 1 : 0 : -1])
        den = 1 - np.dot(phi[1:k], r[1:k])
        phi_kk = num / den
        phi[1:k] = phi[1:k] - phi_kk * phi[k - 1 : 0 : -1]
        phi[k] = phi_kk
        out[k] = phi_kk
    return out


def difference(y: np.ndarray, lag: int = 1) -> np.ndarray:
    """∇_lag y_t = y_t − y_{t−lag}. lag = 1 quita tendencia; lag = periodo quita estacionalidad."""
    y = np.asarray(y, dtype=float)
    return y[lag:] - y[:-lag]


def undifference(diffs: np.ndarray, history: np.ndarray, lag: int = 1) -> np.ndarray:
    """Invierte `difference` sobre pronósticos: y_{t} = ∇y_t + y_{t−lag}, encadenando."""
    out = list(np.asarray(history, dtype=float)[-lag:])
    for d in diffs:
        out.append(d + out[-lag])
    return np.array(out[lag:])
