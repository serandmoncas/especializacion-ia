"""Modelos de pronóstico: baselines, Holt–Winters y AR por mínimos cuadrados."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

# --- Baselines: siempre comparar contra ellos ------------------------------------------------


def naive(y: np.ndarray, h: int) -> np.ndarray:
    """Repite el último valor. Óptimo para una caminata aleatoria."""
    return np.full(h, float(y[-1]))


def seasonal_naive(y: np.ndarray, h: int, m: int) -> np.ndarray:
    """Repite el último ciclo completo: ŷ_{T+k} = y_{T+k−m·⌈k/m⌉}."""
    y = np.asarray(y, dtype=float)
    last = y[-m:]
    return np.array([last[k % m] for k in range(h)])


def drift(y: np.ndarray, h: int) -> np.ndarray:
    """Recta entre la primera y la última observación, extrapolada."""
    y = np.asarray(y, dtype=float)
    slope = (y[-1] - y[0]) / (len(y) - 1)
    return y[-1] + slope * np.arange(1, h + 1)


# --- Suavizamiento exponencial ---------------------------------------------------------------


@dataclass
class HoltWintersFit:
    alpha: float
    beta: float | None
    gamma: float | None
    level: float
    trend: float
    season: np.ndarray  # últimos m índices estacionales
    sse: float

    def forecast(self, h: int) -> np.ndarray:
        k = np.arange(1, h + 1)
        out = np.full(h, self.level, dtype=float)  # sin tendencia ni estación: pronóstico plano
        if self.beta is not None:
            out = out + k * self.trend
        if self.gamma is not None:
            m = len(self.season)
            out = out + self.season[(k - 1) % m]
        return out


def _holt_winters_pass(y, alpha, beta, gamma, m, l0, b0, s0):
    """Recorre la serie con las ecuaciones de error de un paso; devuelve SSE y estados finales."""
    level, trend = l0, b0
    season = list(s0)
    sse = 0.0
    for t, yt in enumerate(y):
        s = season[t] if gamma is not None else 0.0
        pred = level + (trend if beta is not None else 0.0) + s
        err = yt - pred
        sse += err * err
        new_level = alpha * (yt - s) + (1 - alpha) * (level + (trend if beta is not None else 0.0))
        if beta is not None:
            trend = beta * (new_level - level) + (1 - beta) * trend
        if gamma is not None:
            season.append(gamma * (yt - new_level) + (1 - gamma) * s)
        level = new_level
    return sse, level, trend, np.array(season[-m:]) if gamma is not None else np.array([])


def holt_winters(
    y: np.ndarray, *, trend: bool = True, seasonal_period: int | None = None
) -> HoltWintersFit:
    """Suavizamiento exponencial aditivo: simple (sin tendencia ni estación), Holt (tendencia)
    o Holt–Winters (tendencia + estación). Parámetros por mínimo SSE de errores a un paso.

        nivel:     ℓ_t = α(y_t − s_{t−m}) + (1−α)(ℓ_{t−1} + b_{t−1})
        tendencia: b_t = β(ℓ_t − ℓ_{t−1}) + (1−β) b_{t−1}
        estación:  s_t = γ(y_t − ℓ_t) + (1−γ) s_{t−m}

    Estados iniciales: nivel = media del primer ciclo, tendencia = cambio medio entre los dos
    primeros ciclos, estacionalidad = desvío medio respecto de la media de su ciclo, promediado
    sobre los primeros ciclos (hasta 4). Inicializar con *un* solo ciclo ruidoso hace que el
    optimizador lleve γ al borde (γ → 1) para sobrescribirlo cuanto antes, y el modelo degenera en
    “copiar el último ciclo”.
    """
    y = np.asarray(y, dtype=float)
    m = seasonal_period or 1
    use_season = seasonal_period is not None
    if use_season and len(y) < 2 * m:
        raise ValueError("Se necesitan al menos dos ciclos completos")
    l0 = y[:m].mean() if use_season else y[0]
    b0 = ((y[m : 2 * m].mean() - y[:m].mean()) / m) if use_season else (y[1] - y[0])
    if use_season:
        k = min(4, len(y) // m)
        cycles = y[: k * m].reshape(k, m)
        s0 = (cycles - cycles.mean(axis=1, keepdims=True)).mean(axis=0)
    else:
        s0 = np.array([])

    names = ["alpha"] + (["beta"] if trend else []) + (["gamma"] if use_season else [])

    def unpack(p):
        d = dict(zip(names, p, strict=True))
        return d["alpha"], d.get("beta"), d.get("gamma")

    def objective(p):
        a, b, g = unpack(p)
        return _holt_winters_pass(y, a, b, g, m, l0, b0, s0)[0]

    x0 = [0.3] + ([0.05] if trend else []) + ([0.1] if use_season else [])
    res = minimize(objective, x0, bounds=[(1e-4, 1 - 1e-4)] * len(names), method="L-BFGS-B")
    a, b, g = unpack(res.x)
    sse, level, tr, season = _holt_winters_pass(y, a, b, g, m, l0, b0, s0)
    return HoltWintersFit(a, b, g, level, tr if trend else 0.0, season, sse)


# --- Autorregresivo --------------------------------------------------------------------------


@dataclass
class ARFit:
    intercept: float
    coefs: np.ndarray  # φ_1 … φ_p
    sigma2: float

    def forecast(self, history: np.ndarray, h: int) -> np.ndarray:
        """Pronóstico recursivo: cada predicción alimenta los rezagos de la siguiente."""
        buf = list(np.asarray(history, dtype=float)[-len(self.coefs) :])
        out = []
        for _ in range(h):
            nxt = self.intercept + np.dot(self.coefs, buf[::-1][: len(self.coefs)])
            out.append(nxt)
            buf.append(nxt)
        return np.array(out)


def fit_ar(y: np.ndarray, p: int) -> ARFit:
    """AR(p) por mínimos cuadrados condicionales: regresar y_t sobre (1, y_{t−1}, …, y_{t−p})."""
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(y) - p)] + [y[p - k : len(y) - k] for k in range(1, p + 1)])
    target = y[p:]
    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    resid = target - X @ beta
    return ARFit(float(beta[0]), beta[1:], float(resid @ resid / (len(target) - p - 1)))
