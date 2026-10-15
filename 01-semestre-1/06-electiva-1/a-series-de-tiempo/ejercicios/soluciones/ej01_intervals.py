"""Solución ejercicio 1: intervalos de predicción de un AR(1).

y_{T+h} − ŷ_{T+h} = Σ_{j<h} φ^j ε_{T+h−j}, así que Var = σ² Σ_{j<h} φ^{2j}, que crece hacia la
varianza incondicional σ²/(1 − φ²). La fórmula supone φ y σ conocidos: con pocos datos, la
incertidumbre de la estimación hace que el intervalo nominal del 95 % cubra menos.
"""

import numpy as np
from applied_ts.models import fit_ar


def ar1_interval(
    fit, history: np.ndarray, h: int, z: float = 1.96
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(pronóstico, límite inferior, límite superior) para h = 1…h."""
    point = fit.forecast(history, h)
    phi = fit.coefs[0]
    var = fit.sigma2 * np.cumsum(phi ** (2 * np.arange(h)))
    half = z * np.sqrt(var)
    return point, point - half, point + half


def coverage(
    n_train: int, h: int = 10, phi: float = 0.8, reps: int = 2000, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hits = np.zeros(h)
    for _ in range(reps):
        y = np.zeros(n_train + h + 50)
        for t in range(1, len(y)):
            y[t] = 1.0 + phi * y[t - 1] + rng.normal()
        y = y[50:]
        train, test = y[:n_train], y[n_train:]
        _, lo, hi = ar1_interval(fit_ar(train, 1), train, h)
        hits += (test >= lo) & (test <= hi)
    return hits / reps


def check() -> None:
    big = coverage(1000)
    small = coverage(30)
    assert np.all(np.abs(big - 0.95) < 0.025), big
    assert small.mean() < big.mean(), "con pocos datos la cobertura debería caer"
    print(
        f"ej01 ok — cobertura media: n = 1000 → {big.mean():.3f}; n = 30 → {small.mean():.3f}"
        f" (h = 10: {small[-1]:.3f})"
    )


if __name__ == "__main__":
    check()
