"""Mini-proyecto: pronóstico horario del día siguiente de alquileres de bicicletas (UCI).

Cada medianoche de jul–dic 2012 (184 orígenes) se pronostican las 24 horas siguientes con
baselines estacionales, Holt–Winters propio y gradient boosting sobre rezagos. Evaluación por
origen móvil, excluyendo las horas de cierre del sistema (huracán Sandy).

Uso:
    uv run python 01-semestre-1/06-electiva-1/a-series-de-tiempo/data/download.py
    uv run python 01-semestre-1/06-electiva-1/a-series-de-tiempo/proyecto/run.py   # ~2 min
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from applied_ts.bikes import load_hourly
from applied_ts.evaluation import mae, rmse
from applied_ts.features import calendar_features, lag_features
from applied_ts.models import holt_winters, seasonal_naive
from sklearn.ensemble import HistGradientBoostingRegressor

from common import data_dir, save_fig, set_seed, setup_style

MODULE = "01-semestre-1/06-electiva-1/a-series-de-tiempo"
OUT = Path(__file__).resolve().parent / "resultados"
SEED = 42
H = 24
TEST_START, TEST_END = "2012-07-01", "2012-12-31"
LAGS = [24, 48, 72, 96, 120, 144, 168, 336, 504]
WEATHER = ["temp", "atemp", "hum", "windspeed", "weathersit"]


def build_features(df: pd.DataFrame, *, weather: bool) -> pd.DataFrame:
    y = df["cnt"].astype(float)
    parts = [
        lag_features(y, LAGS, rolling=[24, 168], min_lag=24),
        calendar_features(df.index),
        df[["holiday"]],
    ]
    if weather:
        parts.append(df[WEATHER])
    return pd.concat(parts, axis=1)


def prediction_interval(
    residuals: pd.DataFrame, level: float = 0.8
) -> tuple[np.ndarray, np.ndarray]:
    """Intervalo de predicción para las 24 horas del día siguiente.

    `residuals` tiene una fila por día ya pronosticado (antes del origen actual) y una columna
    por hora (0–23), con residuos = real − pronóstico del modelo. Devuelve (desplazamiento
    inferior, desplazamiento superior) de largo 24, a sumar al pronóstico puntual.
    """
    # TODO: implementar (5-10 líneas). Ver README del proyecto, sección "Tu turno".
    raise NotImplementedError


def main() -> None:
    set_seed(SEED)
    setup_style()
    OUT.mkdir(exist_ok=True)
    df = load_hourly(data_dir(MODULE))
    y = df["cnt"].astype(float)
    origins = pd.date_range(TEST_START, TEST_END, freq="D")
    feats = {w: build_features(df, weather=w) for w in (False, True)}

    preds: dict[str, list[np.ndarray]] = {
        k: []
        for k in [
            "ayer",
            "semana_pasada",
            "media_4_semanas",
            "holt_winters_168",
            "hgb_rezagos",
            "hgb_rezagos_clima",
        ]
    }
    timings = dict.fromkeys(preds, 0.0)
    models: dict[bool, HistGradientBoostingRegressor] = {}
    trained_month = None

    for origin in origins:
        hist = y[: origin - pd.Timedelta(hours=1)].to_numpy()
        target_idx = pd.date_range(origin, periods=H, freq="h")

        t0 = time.perf_counter()
        preds["ayer"].append(seasonal_naive(hist, H, 24))
        timings["ayer"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        preds["semana_pasada"].append(seasonal_naive(hist, H, 168))
        timings["semana_pasada"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        weeks = np.stack([hist[len(hist) - 168 * w : len(hist) - 168 * w + H] for w in range(1, 5)])
        preds["media_4_semanas"].append(weeks.mean(axis=0))
        timings["media_4_semanas"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        hw = holt_winters(hist[-168 * 8 :], trend=False, seasonal_period=168)
        preds["holt_winters_168"].append(np.maximum(hw.forecast(H), 0))
        timings["holt_winters_168"] += time.perf_counter() - t0

        # Gradient boosting: se reentrena al empezar cada mes con toda la historia disponible.
        if origin.month != trained_month:
            for w in (False, True):
                t0 = time.perf_counter()
                X = feats[w][: origin - pd.Timedelta(hours=1)].dropna()
                train_y = y.loc[X.index]
                keep = ~df.loc[X.index, "cierre"]  # no aprender de horas de cierre
                models[w] = HistGradientBoostingRegressor(
                    max_iter=400, learning_rate=0.05, random_state=SEED
                ).fit(X[keep], train_y[keep])
                timings["hgb_rezagos_clima" if w else "hgb_rezagos"] += time.perf_counter() - t0
            trained_month = origin.month
        for w, name in ((False, "hgb_rezagos"), (True, "hgb_rezagos_clima")):
            t0 = time.perf_counter()
            preds[name].append(np.maximum(models[w].predict(feats[w].loc[target_idx]), 0))
            timings[name] += time.perf_counter() - t0

    actual = np.stack([y[pd.date_range(o, periods=H, freq="h")].to_numpy() for o in origins])
    closed = np.stack(
        [df["cierre"][pd.date_range(o, periods=H, freq="h")].to_numpy() for o in origins]
    )
    valid = ~closed
    train_end = pd.Timestamp(TEST_START) - pd.Timedelta(hours=1)
    y_train = y[:train_end].to_numpy()
    scale = np.mean(np.abs(y_train[168:] - y_train[:-168]))  # MAE in-sample del naive semanal

    results = {}
    for name, p in preds.items():
        P = np.stack(p)
        results[name] = {
            "mae": mae(actual[valid], P[valid]),
            "rmse": rmse(actual[valid], P[valid]),
            "mase_168": mae(actual[valid], P[valid]) / scale,
            "seconds": timings[name],
        }
        preds[name] = P
    best_baseline = min(
        ("ayer", "semana_pasada", "media_4_semanas", "holt_winters_168"),
        key=lambda n: results[n]["mae"],
    )

    # Comparación pareada por día: MAE diario del modelo vs. el mejor baseline
    def daily_mae(P):
        return np.array(
            [
                np.mean(np.abs(actual[d][valid[d]] - P[d][valid[d]])) if valid[d].any() else np.nan
                for d in range(len(origins))
            ]
        )

    rng = np.random.default_rng(SEED)
    base_daily = daily_mae(preds[best_baseline])
    paired = {}
    for name in ("hgb_rezagos", "hgb_rezagos_clima"):
        diff = daily_mae(preds[name]) - base_daily
        diff = diff[~np.isnan(diff)]
        boots = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
        paired[name] = {
            "dias_mejor_que_baseline": float(np.mean(diff < 0)),
            "diferencia_media_mae": float(diff.mean()),
            "ic95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        }

    # Intervalos de predicción (si están implementados): residuos de días anteriores al origen
    coverage = None
    try:
        p_best = preds["hgb_rezagos"]
        res = pd.DataFrame(actual - p_best)
        hits, widths = [], []
        for d in range(28, len(origins)):  # 4 semanas de residuos para calibrar
            lo, hi = prediction_interval(res.iloc[:d], 0.8)
            ok = valid[d]
            hits.extend(((actual[d] >= p_best[d] + lo) & (actual[d] <= p_best[d] + hi))[ok])
            widths.extend((hi - lo)[ok])
        coverage = {
            "nominal": 0.8,
            "empirica": float(np.mean(hits)),
            "ancho_medio": float(np.mean(widths)),
        }
    except NotImplementedError:
        print("prediction_interval() sin implementar: se omite la evaluación de intervalos")

    report = {
        "origenes": len(origins),
        "horas_excluidas_por_cierre": int(closed.sum()),
        "escala_mase": float(scale),
        "resultados": results,
        "mejor_baseline": best_baseline,
        "comparacion_diaria_vs_mejor_baseline": paired,
        "intervalos": coverage,
    }
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # --- Figuras ---------------------------------------------------------------------------
    week = slice(70, 77)  # una semana de septiembre
    idx = pd.date_range(origins[week.start], periods=7 * H, freq="h")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(idx, actual[week].ravel(), color="k", lw=1.5, label="real")
    for name in (best_baseline, "hgb_rezagos", "hgb_rezagos_clima"):
        ax.plot(idx, preds[name][week].ravel(), lw=1, label=name)
    ax.set(
        ylabel="alquileres por hora",
        title="Pronóstico del día siguiente: una semana de septiembre 2012",
    )
    ax.legend(fontsize=8, ncol=4)
    save_fig(fig, OUT / "semana.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    for name in preds:
        err = np.where(valid, np.abs(actual - preds[name]), np.nan)
        ax.plot(range(H), np.nanmean(err, axis=0), "o-", ms=3, label=name)
    ax.set(xlabel="hora del día", ylabel="MAE", title="Error por hora del día (184 días)")
    ax.legend(fontsize=7)
    save_fig(fig, OUT / "error_por_hora.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    names = sorted(results, key=lambda n: results[n]["mase_168"])
    ax.barh(names[::-1], [results[n]["mase_168"] for n in names[::-1]])
    ax.axvline(1, color="k", ls="--", lw=0.8)
    ax.set(xlabel="MASE (1 = naive semanal in-sample)", title="Comparación de modelos")
    save_fig(fig, OUT / "mase.png")

    print(f"{len(origins)} orígenes, {int(closed.sum())} horas de cierre excluidas")
    print(f"{'modelo':<20}{'MAE':>8}{'RMSE':>8}{'MASE':>7}{'seg':>7}")
    for name in names:
        r = results[name]
        print(
            f"{name:<20}{r['mae']:>8.1f}{r['rmse']:>8.1f}{r['mase_168']:>7.3f}{r['seconds']:>7.1f}"
        )
    print(f"mejor baseline: {best_baseline}")
    for name, p in paired.items():
        print(
            f"{name}: mejor en {p['dias_mejor_que_baseline']:.0%} de los días,"
            f" ΔMAE {p['diferencia_media_mae']:.1f}"
            f" [{p['ic95'][0]:.1f}, {p['ic95'][1]:.1f}]"
        )
    if coverage:
        print(
            f"intervalos 80 %: cobertura {coverage['empirica']:.3f},"
            f" ancho {coverage['ancho_medio']:.0f}"
        )


if __name__ == "__main__":
    main()
