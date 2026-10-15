"""Métricas de pronóstico y validación por origen móvil (backtesting)."""

from collections.abc import Callable, Iterator

import numpy as np


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def smape(y, yhat) -> float:
    """MAPE simétrico en %. Evita dividir por cero cuando y = 0 (si también ŷ = 0, aporta 0)."""
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    den = np.abs(y) + np.abs(yhat)
    return float(
        100 * np.mean(np.where(den == 0, 0.0, 2 * np.abs(y - yhat) / np.where(den == 0, 1, den)))
    )


def mase(y, yhat, y_train, m: int = 1) -> float:
    """MAE escalado por el MAE in-sample del seasonal naive de periodo m (Hyndman & Koehler, 2006).

    < 1: mejor que repetir el ciclo anterior. Comparable entre series de escalas distintas y
    bien definido aunque haya ceros (a diferencia del MAPE).
    """
    y_train = np.asarray(y_train, dtype=float)
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return mae(y, yhat) / scale


def rolling_origin(
    n: int, *, initial: int, horizon: int, step: int = 1
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Particiones (train = [0, t), test = [t, t + horizon)) para t = initial, initial + step, …

    A diferencia de k-fold, el futuro nunca se usa para predecir el pasado: así se pronostica
    en producción. Con step = horizon las ventanas de test no se solapan.
    """
    t = initial
    while t + horizon <= n:
        yield np.arange(t), np.arange(t, t + horizon)
        t += step


def backtest(
    y: np.ndarray,
    forecaster: Callable[[np.ndarray, int], np.ndarray],
    *,
    initial: int,
    horizon: int,
    step: int | None = None,
) -> dict[str, np.ndarray]:
    """Aplica `forecaster(historia, h)` en cada origen; devuelve pronósticos y errores por paso."""
    y = np.asarray(y, dtype=float)
    preds, actual = [], []
    for train, test in rolling_origin(
        len(y), initial=initial, horizon=horizon, step=step or horizon
    ):
        preds.append(forecaster(y[train], horizon))
        actual.append(y[test])
    preds, actual = np.array(preds), np.array(actual)
    return {"pred": preds, "actual": actual, "abs_err_by_step": np.abs(preds - actual).mean(axis=0)}
