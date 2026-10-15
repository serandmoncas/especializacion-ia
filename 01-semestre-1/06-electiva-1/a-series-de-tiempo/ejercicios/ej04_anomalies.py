"""Ejercicio 4: detección de anomalías con residuos robustos. Ver README.md."""

import numpy as np
import pandas as pd


def robust_anomalies(y: pd.Series, period: int = 168, weeks: int = 4, k: float = 4.0) -> pd.Series:
    """True donde |y − mediana de las últimas `weeks` semanas a la misma hora| > k · σ̂_MAD."""
    raise NotImplementedError


def check() -> None:
    from applied_ts.bikes import load_hourly

    from common import data_dir

    try:
        df = load_hourly(data_dir("01-semestre-1/06-electiva-1/a-series-de-tiempo"))
    except FileNotFoundError:
        print(
            "ej04: falta data/hour.csv (ejecutar data/download.py); se prueba con datos sintéticos"
        )
        idx = pd.date_range("2024-01-01", periods=168 * 8, freq="h")
        rng = np.random.default_rng(0)
        y = pd.Series(
            100 + 50 * np.sin(2 * np.pi * np.arange(len(idx)) / 24) + rng.normal(0, 5, len(idx)),
            idx,
        )
        y.iloc[-30:-10] = 0
        flags = robust_anomalies(y)
        assert flags.iloc[-30:-10].mean() > 0.8 and flags.iloc[:-30].mean() < 0.02
        print("ej04 ok (sintético)")
        return
    flags = robust_anomalies(df["cnt"].astype(float))
    hours = df.index.hour
    day_closure = df["cierre"] & (hours >= 7) & (hours <= 22)
    caught_day = flags[day_closure].mean()
    caught_night = flags[df["cierre"] & ~day_closure].mean()
    top = flags.groupby(flags.index.date).sum().sort_values(ascending=False).head(5)
    top_days = [str(d) for d in top.index]
    # De noche lo normal ya es casi 0: un cierre no es anómalo. De día sí debe detectarse.
    assert caught_day > 0.5, caught_day
    assert {"2012-10-29", "2012-10-30"} <= set(top_days[:3]), top_days
    print(
        f"ej04 ok — detecta el {caught_day:.0%} de las horas de cierre diurnas"
        f" (nocturnas: {caught_night:.0%}); días más anómalos: {top_days}"
    )


if __name__ == "__main__":
    check()
