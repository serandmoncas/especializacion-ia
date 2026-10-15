"""Mini-proyecto: pipeline completo de clasificación sobre UCI Adult (ingreso > 50K).

1. Selección de modelo con validación cruzada anidada (sin sesgo optimista).
2. Comparación desde cero vs. scikit-learn con los mismos hiperparámetros.
3. Evaluación única en el test oficial: AUC con IC bootstrap, calibración, umbral.
4. Importancia por permutación y métricas por subgrupo (antesala del módulo de ética).

Uso:
    uv run python 01-semestre-1/04-aprendizaje-automatico/data/download.py
    uv run python 01-semestre-1/04-aprendizaje-automatico/proyecto/run.py   # ~3 min
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from applied_ml import DecisionTreeClassifier, LogisticRegression
from applied_ml.adult import TARGET, load_adult
from applied_ml.metrics import (
    average_precision,
    brier_score,
    calibration_bins,
    expected_calibration_error,
    log_loss,
    precision_recall_curve,
    precision_recall_f1,
    roc_auc,
    roc_curve,
)
from applied_ml.model_selection import bootstrap_metric_ci, stratified_kfold
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import FunctionTransformer, Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier as SkDecisionTree

from common import data_dir, save_fig, set_seed, setup_style

MODULE = "01-semestre-1/04-aprendizaje-automatico"
OUT = Path(__file__).resolve().parent / "resultados"
SEED = 42


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ingeniería de features sobre las columnas crudas (se aplica dentro del Pipeline).

    Las columnas numéricas nuevas se escalan y las de texto se codifican one-hot de forma
    automática (el preprocesador selecciona por tipo). Lo que se agregue aquí se evalúa con la
    misma validación cruzada anidada, así que su efecto es medible.
    """
    out = df.copy()
    # TODO: agregar features (5-10 líneas). Ver README del proyecto, sección "Tu turno".
    return out


def build_preprocessor() -> Pipeline:
    # Sin log1p en capital_gain/loss a propósito: se probó y baja el AUC de la logística en los
    # 5 folds (0.907 → 0.900). La relación es casi un umbral (ganancias grandes ⇒ >50K) y la
    # escala log aplasta justo esos valores. Ver README, “Lo que no funcionó”.
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="Missing"),
        OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False),
    )
    columns = ColumnTransformer(
        [
            ("num", numeric, make_column_selector(dtype_include="number")),
            ("cat", categorical, make_column_selector(dtype_exclude="number")),
        ],
        verbose_feature_names_out=False,
    )
    return make_pipeline(FunctionTransformer(engineer_features), columns)


def candidates() -> dict[str, tuple[object, dict]]:
    """Familia de modelo → (estimador, grilla de hiperparámetros para la búsqueda interna)."""
    return {
        "baseline": (DummyClassifier(strategy="prior"), {}),
        "logistic": (SkLogisticRegression(max_iter=2000), {"model__C": [0.01, 0.1, 1.0]}),
        "random_forest": (
            RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
            {"model__min_samples_leaf": [2, 10], "model__max_features": ["sqrt", 0.3]},
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(random_state=SEED),
            {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31, 63]},
        ),
    }


def make_pipe(model) -> Pipeline:
    return Pipeline([("pre", build_preprocessor()), ("model", model)])


def nested_cv(X: pd.DataFrame, y: np.ndarray, rng: np.random.Generator) -> dict:
    """CV externa (5 folds) estima el desempeño; CV interna (3 folds) elige hiperparámetros.

    El mejor puntaje de la búsqueda interna es optimista: es el máximo de varias estimaciones
    ruidosas. El puntaje externo, en datos que la búsqueda nunca vio, no lo es.
    """
    outer = stratified_kfold(y, 5, rng)
    results = {}
    for name, (model, grid) in candidates().items():
        t0 = time.perf_counter()
        outer_scores, inner_best, chosen = [], [], []
        for train, test in outer:
            search = GridSearchCV(
                make_pipe(model),
                grid or {"model__strategy": ["prior"]},
                scoring="roc_auc",
                cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                n_jobs=1 if name == "random_forest" else -1,
            )
            search.fit(X.iloc[train], y[train])
            proba = search.predict_proba(X.iloc[test])[:, 1]
            outer_scores.append(roc_auc(y[test], proba))
            inner_best.append(search.best_score_)
            chosen.append({k.removeprefix("model__"): v for k, v in search.best_params_.items()})
        results[name] = {
            "outer_auc": outer_scores,
            "outer_mean": float(np.mean(outer_scores)),
            "outer_std": float(np.std(outer_scores, ddof=1)),
            "inner_best_mean": float(np.mean(inner_best)),
            "chosen_params": chosen,
            "seconds": time.perf_counter() - t0,
        }
        print(
            f"  {name:<24} AUC externa {results[name]['outer_mean']:.4f}"
            f" ± {results[name]['outer_std']:.4f}"
            f"  (interna {results[name]['inner_best_mean']:.4f})  {results[name]['seconds']:.0f}s"
        )
    return results


def scratch_vs_library(X_tr, y_tr, X_te, y_te) -> dict:
    """Mismos hiperparámetros, mismo preprocesamiento: ¿el código propio da el mismo modelo?"""
    pairs = {
        "logistic (C=1)": (
            SkLogisticRegression(C=1.0, max_iter=5000, tol=1e-10),
            LogisticRegression(alpha=1 / len(y_tr), solver="newton"),
        ),
        "árbol (prof. 8, hoja ≥ 20)": (
            SkDecisionTree(max_depth=8, min_samples_leaf=20, random_state=SEED),
            DecisionTreeClassifier(max_depth=8, min_samples_leaf=20),
        ),
    }
    out = {}
    for name, (lib, ours) in pairs.items():
        row = {}
        for who, model in (("sklearn", lib), ("desde_cero", ours)):
            pipe = make_pipe(model)
            t0 = time.perf_counter()
            pipe.fit(X_tr, y_tr)
            row[who] = {
                "auc": roc_auc(y_te, pipe.predict_proba(X_te)[:, 1]),
                "seconds": time.perf_counter() - t0,
            }
        out[name] = row
    return out


def choose_threshold_oof(pipe, X, y) -> tuple[float, np.ndarray]:
    """Umbral que maximiza F1 en predicciones out-of-fold de *train*: el test no se toca."""
    oof = cross_val_predict(
        pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba"
    )[:, 1]
    precision, recall, thr = precision_recall_curve(y, oof)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thr[np.argmax(f1)]), oof


def subgroup_report(
    df: pd.DataFrame, y: np.ndarray, proba: np.ndarray, threshold: float, col: str
) -> dict:
    out = {}
    pred = proba >= threshold
    for g in sorted(df[col].dropna().unique()):
        m = (df[col] == g).to_numpy()
        yt, pt = y[m], pred[m]
        out[str(g)] = {
            "n": int(m.sum()),
            "prevalencia": float(yt.mean()),
            "tasa_predicha_positiva": float(pt.mean()),
            "auc": roc_auc(yt, proba[m]),
            "tpr": float(pt[yt == 1].mean()),
            "fpr": float(pt[yt == 0].mean()),
        }
    return out


def main() -> None:
    rng = set_seed(SEED)
    setup_style()
    OUT.mkdir(exist_ok=True)
    train, test = load_adult(data_dir(MODULE))
    X_tr, y_tr = train.drop(columns=TARGET), train[TARGET].to_numpy()
    X_te, y_te = test.drop(columns=TARGET), test[TARGET].to_numpy()

    print("Validación cruzada anidada (train oficial, 32 561 filas):")
    ncv = nested_cv(X_tr, y_tr, rng)
    best_name = max((n for n in ncv if n != "baseline"), key=lambda n: ncv[n]["outer_mean"])

    print("\nDesde cero vs. sklearn (train → test):")
    svl = scratch_vs_library(X_tr, y_tr, X_te, y_te)
    for name, row in svl.items():
        print(
            f"  {name:<28} sklearn AUC {row['sklearn']['auc']:.4f}"
            f" ({row['sklearn']['seconds']:.2f}s) | desde cero AUC {row['desde_cero']['auc']:.4f}"
            f" ({row['desde_cero']['seconds']:.2f}s)"
        )

    # Modelo final: búsqueda interna sobre todo train, luego una sola evaluación en test.
    model, grid = candidates()[best_name]
    final = GridSearchCV(
        make_pipe(model),
        grid,
        scoring="roc_auc",
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        n_jobs=-1,
    ).fit(X_tr, y_tr)
    threshold, _ = choose_threshold_oof(final.best_estimator_, X_tr, y_tr)
    proba = final.predict_proba(X_te)[:, 1]
    logistic_proba = (
        make_pipe(SkLogisticRegression(C=1.0, max_iter=2000))
        .fit(X_tr, y_tr)
        .predict_proba(X_te)[:, 1]
    )

    auc, auc_lo, auc_hi = bootstrap_metric_ci(roc_auc, y_te, proba, n_boot=2000, rng=rng)
    p_thr, r_thr, f1_thr = precision_recall_f1(y_te, proba >= threshold)
    p_05, r_05, f1_05 = precision_recall_f1(y_te, proba >= 0.5)
    test_metrics = {
        "model": best_name,
        "params": final.best_params_,
        "auc": auc,
        "auc_ci95": [auc_lo, auc_hi],
        "average_precision": average_precision(y_te, proba),
        "log_loss": log_loss(y_te, proba),
        "brier": brier_score(y_te, proba),
        "ece": expected_calibration_error(y_te, proba),
        "threshold_f1_oof": threshold,
        "at_threshold": {"precision": p_thr, "recall": r_thr, "f1": f1_thr},
        "at_0.5": {"precision": p_05, "recall": r_05, "f1": f1_05},
        "accuracy_baseline": float(1 - y_te.mean()),
        "accuracy": float(np.mean((proba >= threshold) == y_te)),
    }

    perm = permutation_importance(
        final.best_estimator_,
        X_te,
        y_te,
        scoring="roc_auc",
        n_repeats=5,
        random_state=SEED,
        n_jobs=-1,
    )
    importance = dict(
        sorted(
            zip(X_te.columns, perm.importances_mean.tolist(), strict=True), key=lambda kv: -kv[1]
        )
    )
    subgroups = {col: subgroup_report(X_te, y_te, proba, threshold, col) for col in ("sex", "race")}

    report = {
        "nested_cv": ncv,
        "best_model": best_name,
        "scratch_vs_library": svl,
        "test": test_metrics,
        "permutation_importance_auc": importance,
        "subgroups": subgroups,
    }
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))

    # --- Figuras -----------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [n for n in ncv if n != "baseline"]
    ax.boxplot([ncv[n]["outer_auc"] for n in names], tick_labels=names)
    ax.scatter(
        range(1, len(names) + 1),
        [ncv[n]["inner_best_mean"] for n in names],
        marker="x",
        color="C3",
        zorder=3,
        label="mejor puntaje de la búsqueda interna",
    )
    ax.set_ylabel("AUC en folds externos")
    ax.set_title("Validación cruzada anidada")
    ax.legend(loc="lower right")
    save_fig(fig, OUT / "nested_cv.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for p, label in ((proba, best_name), (logistic_proba, "logistic")):
        fpr, tpr, _ = roc_curve(y_te, p)
        axes[0].plot(fpr, tpr, label=f"{label} (AUC {roc_auc(y_te, p):.3f})")
        prec, rec, _ = precision_recall_curve(y_te, p)
        axes[1].plot(rec, prec, label=f"{label} (AP {average_precision(y_te, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[1].axhline(y_te.mean(), color="k", ls="--", lw=0.8, label="azar (prevalencia)")
    axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC (test)")
    axes[1].set(xlabel="recall", ylabel="precisión", title="Precisión–recall (test)")
    for ax in axes:
        ax.legend(fontsize=8)
    save_fig(fig, OUT / "roc_pr.png")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for p, label in ((proba, best_name), (logistic_proba, "logistic")):
        mp, fr, _ = calibration_bins(y_te, p, n_bins=10)
        ax.plot(mp, fr, "o-", label=f"{label} (ECE {expected_calibration_error(y_te, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(
        xlabel="probabilidad predicha", ylabel="frecuencia observada", title="Calibración (test)"
    )
    ax.legend()
    save_fig(fig, OUT / "calibracion.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    items = list(importance.items())[::-1]
    ax.barh([k for k, _ in items], [v for _, v in items])
    ax.set_xlabel("caída de AUC al permutar la columna")
    ax.set_title(f"Importancia por permutación ({best_name}, test)")
    save_fig(fig, OUT / "importancia.png")

    print(f"\nModelo final: {best_name} {final.best_params_}")
    print(
        f"Test: AUC {auc:.4f} [{auc_lo:.4f}, {auc_hi:.4f}]"
        f"  AP {test_metrics['average_precision']:.4f}"
        f"  ECE {test_metrics['ece']:.4f}"
    )
    print(
        f"Umbral OOF (máx. F1) = {threshold:.3f}: P {p_thr:.3f} R {r_thr:.3f} F1 {f1_thr:.3f}"
        f"  | umbral 0.5: P {p_05:.3f} R {r_05:.3f} F1 {f1_05:.3f}"
    )
    print(
        "Por sexo:",
        {g: {k: round(v, 3) for k, v in r.items()} for g, r in subgroups["sex"].items()},
    )


if __name__ == "__main__":
    main()
