"""Ejercicio 3: ARMA(1,1) por mínimos cuadrados condicionales. Ver README.md."""

import numpy as np


def arma11_residuals(params: np.ndarray, y: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def fit_arma11(y: np.ndarray) -> np.ndarray:
    """Devuelve (c, φ, θ)."""
    raise NotImplementedError


def check() -> None:
    from statsmodels.tsa.arima.model import ARIMA

    rng = np.random.default_rng(0)
    n, c, phi, theta = 3000, 2.0, 0.7, 0.4
    e = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = c + phi * y[t - 1] + e[t] + theta * e[t - 1]
    ours = fit_arma11(y)
    ref = ARIMA(y, order=(1, 0, 1)).fit()
    assert abs(ours[1] - phi) < 0.05 and abs(ours[2] - theta) < 0.05, ours
    ref_phi, ref_theta = ref.params[1], ref.params[2]
    assert abs(ours[1] - ref_phi) < 0.02 and abs(ours[2] - ref_theta) < 0.02
    print(
        f"ej03 ok — CSS: φ = {ours[1]:.3f}, θ = {ours[2]:.3f}"
        f" | statsmodels: φ = {ref_phi:.3f}, θ = {ref_theta:.3f}"
    )


if __name__ == "__main__":
    check()
