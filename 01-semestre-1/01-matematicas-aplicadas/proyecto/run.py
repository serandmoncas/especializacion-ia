"""Mini-proyecto: regresión lineal desde cero sobre California Housing.

Compara forma cerrada, GD, GD + momentum y SGD por minilotes contra scikit-learn,
y guarda métricas y figuras en `resultados/`.

Uso:
    uv run python 01-semestre-1/01-matematicas-aplicadas/data/download.py
    uv run python 01-semestre-1/01-matematicas-aplicadas/proyecto/run.py
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from applied_math import LinearRegressionGD, fit_normal_equation
from applied_math.linear_regression import add_bias
from sklearn.linear_model import LinearRegression

from common import data_dir, save_fig, set_seed, setup_style

MODULE = "01-semestre-1/01-matematicas-aplicadas"
OUT = Path(__file__).resolve().parent / "resultados"
TARGET = "MedHouseVal"
SEED = 42


def load_split(test_size: float = 0.2) -> tuple[np.ndarray, ...]:
    csv = data_dir(MODULE) / "california_housing.csv"
    if not csv.exists():
        raise SystemExit(f"Falta {csv}. Ejecutar primero data/download.py")
    df = pd.read_csv(csv)
    X, y = df.drop(columns=TARGET).to_numpy(), df[TARGET].to_numpy()
    idx = np.random.default_rng(SEED).permutation(len(df))
    n_test = int(len(df) * test_size)
    test, train = idx[:n_test], idx[n_test:]
    return X[train], X[test], y[train], y[test], list(df.drop(columns=TARGET).columns)


def standardize(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Estadísticos solo de train: usar los de test sería fuga de información.
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
    return (X_train - mu) / sd, (X_test - mu) / sd


def metrics(y: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    r2 = float(1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2))
    return {"rmse": rmse, "r2": r2}


def main() -> None:
    set_seed(SEED)
    setup_style()
    X_tr, X_te, y_tr, y_te, features = load_split()
    X_tr, X_te = standardize(X_tr, X_te)

    # El lr seguro para GD se deriva de la curvatura máxima: η < 2/L, L = λ_max(2XᵀX/n).
    Xb = add_bias(X_tr)
    lams = np.linalg.eigvalsh(2 * Xb.T @ Xb / len(Xb))
    L, kappa = lams[-1], lams[-1] / lams[0]
    lr = 1.0 / L

    models = {
        "gd": LinearRegressionGD(solver="gd", lr=lr, max_iter=20_000),
        "gd_momentum": LinearRegressionGD(solver="gd", lr=lr, momentum=0.9, max_iter=20_000),
        "sgd_b64": LinearRegressionGD(solver="sgd", lr=0.01, batch_size=64, epochs=30, seed=SEED),
    }

    results: dict[str, dict] = {}
    t0 = time.perf_counter()
    w = fit_normal_equation(X_tr, y_tr)
    results["normal_eq"] = {
        **metrics(y_te, X_te @ w[1:] + w[0]),
        "seconds": time.perf_counter() - t0,
        "iterations": None,
    }

    t0 = time.perf_counter()
    sk = LinearRegression().fit(X_tr, y_tr)
    results["sklearn"] = {
        **metrics(y_te, sk.predict(X_te)),
        "seconds": time.perf_counter() - t0,
        "iterations": None,
    }

    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        elapsed = time.perf_counter() - t0
        results[name] = {
            **metrics(y_te, model.predict(X_te)),
            "seconds": elapsed,
            "iterations": model.result_.n_iter,
            "coef_max_abs_diff_vs_normal_eq": float(
                np.abs(np.r_[model.intercept_, model.coef_] - w).max()
            ),
        }

    OUT.mkdir(exist_ok=True)
    summary = {
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "hessian_L": float(L),
        "condition_number": float(kappa),
        "lr_gd": float(lr),
        "coefficients_normal_eq": dict(
            zip(["intercept", *features], w.round(4).tolist(), strict=True)
        ),
        "models": results,
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    fig, ax = plt.subplots()
    L_star = float(np.mean((Xb @ w - y_tr) ** 2))
    for name, model in models.items():
        hist = np.array(model.result_.history) - L_star
        x_axis = np.arange(len(hist))
        if model.solver == "sgd":
            x_axis = x_axis * int(np.ceil(len(X_tr) / model.batch_size))  # épocas → pasos
        ax.loglog(x_axis + 1, np.maximum(hist, 1e-12), label=name)
    ax.set_xlabel("pasos de actualización")
    ax.set_ylabel("MSE_train − MSE*")
    ax.set_title(f"Convergencia en California Housing (κ = {kappa:.0f})")
    ax.legend()
    save_fig(fig, OUT / "convergencia.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    order = np.argsort(np.abs(w[1:]))
    ax.barh(np.array(features)[order], w[1:][order])
    ax.set_title("Coeficientes (features estandarizadas)")
    save_fig(fig, OUT / "coeficientes.png")

    print(f"κ = {kappa:.1f}, η = 1/L = {lr:.4f}")
    print(f"{'modelo':<14}{'RMSE':>8}{'R²':>8}{'iter':>8}{'seg':>9}")
    for name, r in results.items():
        it = "-" if r["iterations"] is None else r["iterations"]
        print(f"{name:<14}{r['rmse']:>8.4f}{r['r2']:>8.4f}{it:>8}{r['seconds']:>9.4f}")


if __name__ == "__main__":
    main()
