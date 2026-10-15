"""Carga de UCI Bike Sharing como serie horaria completa."""

from pathlib import Path

import pandas as pd

OUTAGE_MIN_HOURS = 6


def load_hourly(data_dir: str | Path) -> pd.DataFrame:
    """Serie horaria sin huecos con `cnt` (alquileres), clima, calendario y bandera `cierre`.

    El archivo omite las horas sin alquileres (el mínimo registrado es 1, nunca 0). Hay dos tipos
    de huecos: horas sueltas de madrugada (alquileres = 0 de verdad) y bloques largos que
    coinciden con cierres del sistema por eventos climáticos (huracán Sandy, oct. 2012; huracán
    Irene, ago. 2011; tormenta de nieve, ene. 2011). Ambos se rellenan con 0, que es lo que se
    alquiló, y los bloques ≥ OUTAGE_MIN_HOURS se marcan en `cierre` para excluirlos al evaluar:
    ningún modelo basado en la historia de alquileres puede anticipar un huracán.
    """
    h = pd.read_csv(Path(data_dir) / "hour.csv", parse_dates=["dteday"])
    h.index = h["dteday"] + pd.to_timedelta(h["hr"], unit="h")
    full = pd.date_range(h.index.min(), h.index.max(), freq="h")
    out = h.reindex(full)
    missing = out["cnt"].isna()
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("size")
    out["cierre"] = missing & (run_len >= OUTAGE_MIN_HOURS)
    for col in ("cnt", "casual", "registered"):
        out[col] = out[col].fillna(0).astype(int)
    for col in ("temp", "atemp", "hum", "windspeed", "weathersit"):
        out[col] = out[col].ffill()
    out["holiday"] = out["holiday"].fillna(0).astype(int)
    return out[
        [
            "cnt",
            "casual",
            "registered",
            "temp",
            "atemp",
            "hum",
            "windspeed",
            "weathersit",
            "holiday",
            "cierre",
        ]
    ]
