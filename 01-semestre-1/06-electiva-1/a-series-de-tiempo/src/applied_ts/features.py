"""Features de rezagos para pronosticar con modelos de ML tabulares."""

import numpy as np
import pandas as pd


def lag_features(
    y: pd.Series, lags: list[int], rolling: list[int] | None = None, min_lag: int = 1
) -> pd.DataFrame:
    """Columnas lag_k = y_{t−k} y medias móviles que terminan en t − min_lag.

    Para pronosticar h pasos adelante con un solo modelo (estrategia *directa*), todos los rezagos
    deben ser ≥ h: si el menor rezago es 24, el modelo puede predecir cualquier hora del día
    siguiente usando solo información disponible a medianoche. Con rezagos más cortos habría que
    pronosticar de forma recursiva, realimentando predicciones y acumulando error.
    """
    if min(lags) < min_lag:
        raise ValueError(f"Todos los rezagos deben ser ≥ {min_lag} para no usar información futura")
    out = {f"lag_{k}": y.shift(k) for k in lags}
    for w in rolling or []:
        out[f"roll_mean_{w}"] = y.shift(min_lag).rolling(w).mean()
    return pd.DataFrame(out, index=y.index)


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hora, día de la semana y mes como enteros, más codificación cíclica de la hora
    (sen/cos: las 23 h y las 0 h quedan cerca, cosa que el entero no expresa)."""
    hour = index.hour
    return pd.DataFrame(
        {
            "hour": hour,
            "dayofweek": index.dayofweek,
            "month": index.month,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
        },
        index=index,
    )
