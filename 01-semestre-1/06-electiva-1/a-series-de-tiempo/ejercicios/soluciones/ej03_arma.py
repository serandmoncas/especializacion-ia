"""Solución ejercicio 3: ARMA(1,1) por mínimos cuadrados condicionales (CSS).

Dado θ = (c, φ, θ_ma), los errores se reconstruyen recursivamente:
ε_t = y_t − c − φ y_{t−1} − θ ε_{t−1}, con ε_0 = 0. La suma de cuadrados es no lineal en θ_ma,
así que se minimiza numéricamente. Con |θ_ma| < 1 (invertibilidad) el efecto de ε_0 = 0 se
desvanece; statsmodels usa máxima verosimilitud exacta y coincide para n grande.
"""

import numpy as np
from scipy.optimize import minimize


def arma11_residuals(params: np.ndarray, y: np.ndarray) -> np.ndarray:
    c, phi, theta = params
    eps = np.zeros(len(y))
    for t in range(1, len(y)):
        eps[t] = y[t] - c - phi * y[t - 1] - theta * eps[t - 1]
    return eps[1:]


def fit_arma11(y: np.ndarray) -> np.ndarray:
    """Devuelve (c, φ, θ)."""
    y = np.asarray(y, dtype=float)
    x0 = np.array([y.mean() * 0.5, 0.5, 0.0])

    def sse(p):
        return np.sum(arma11_residuals(p, y) ** 2)

    bounds = [(None, None), (-0.99, 0.99), (-0.99, 0.99)]
    return minimize(sse, x0, method="L-BFGS-B", bounds=bounds).x


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
