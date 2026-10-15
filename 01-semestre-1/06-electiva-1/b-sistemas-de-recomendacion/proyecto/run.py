"""Mini-proyecto: recomendador de películas sobre MovieLens 1M.

A. Predicción de calificaciones (RMSE): media, sesgos, item-kNN, FunkSVD, ALS.
B. Ranking top-10 (NDCG, recall, cobertura, sesgo de popularidad): popularidad, item-kNN,
   PureSVD, ALS implícito y FunkSVD usado para rankear.

Partición temporal por usuario (20 % más reciente a test). Hiperparámetros elegidos con una
validación temporal dentro del train; el test se usa una sola vez.

Uso:
    uv run python 01-semestre-1/06-electiva-1/b-sistemas-de-recomendacion/data/download.py
    uv run python 01-semestre-1/06-electiva-1/b-sistemas-de-recomendacion/proyecto/run.py  # ~3 min
"""

import itertools
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from applied_recsys.baselines import BiasModel, popularity_scores
from applied_recsys.data import load_movielens_1m, random_split, temporal_split, to_csr
from applied_recsys.factorization import ExplicitALS, FunkSVD, ImplicitALS, PureSVD
from applied_recsys.metrics import (
    catalog_coverage,
    mean_popularity_percentile,
    ranking_metrics,
    rmse,
    top_n,
)
from applied_recsys.neighbors import ItemKNN

from common import data_dir, save_fig, set_seed, setup_style

MODULE = "01-semestre-1/06-electiva-1/b-sistemas-de-recomendacion"
OUT = Path(__file__).resolve().parent / "resultados"
SEED = 42
K = 10
CHUNK = 1000


def diversify(scores: np.ndarray, popularity: np.ndarray) -> np.ndarray:
    """Reajusta los puntajes de un lote de usuarios [n_usuarios, n_ítems] para recomendar menos
    lo obvio. `popularity[i]` = número de calificaciones del ítem i en train.

    Debe devolver puntajes de la misma forma; el script mide cuánta precisión se pierde a cambio
    de cuánta cobertura y novedad.
    """
    # TODO: implementar (5-10 líneas). Ver README del proyecto, sección "Tu turno".
    raise NotImplementedError


def evaluate_ranking(score_fn, R_train, relevant: dict[int, set[int]], popularity, n_items) -> dict:
    users = np.array(sorted(relevant))
    recs = np.vstack(
        [
            top_n(score_fn(users[s : s + CHUNK]), R_train[users[s : s + CHUNK]], K)
            for s in range(0, len(users), CHUNK)
        ]
    )
    per_user = ranking_metrics(recs, [relevant[u] for u in users], K)
    return {
        "per_user": per_user,
        "recs": recs,
        "users": users,
        "coverage": catalog_coverage(recs, n_items),
        "popularity_pct": mean_popularity_percentile(recs, popularity),
    }


def relevant_sets(df, threshold: float = 4.0) -> dict[int, set[int]]:
    pos = df[df["rating"] >= threshold]
    return {int(u): set(g["i"].tolist()) for u, g in pos.groupby("u")}


def bootstrap_ci(values: np.ndarray, rng, n: int = 2000) -> list[float]:
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n)]
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main() -> None:
    rng = set_seed(SEED)
    setup_style()
    OUT.mkdir(exist_ok=True)
    ratings, movies = load_movielens_1m(data_dir(MODULE))
    shape = (int(ratings["u"].max()) + 1, int(ratings["i"].max()) + 1)
    train, test = temporal_split(ratings, 0.2)
    fit_tr, val = temporal_split(train, 0.1)  # validación temporal dentro del train
    R, R_fit = to_csr(train, shape), to_csr(fit_tr, shape)
    pop_full = popularity_scores(R)
    report: dict = {"n_train": len(train), "n_test": len(test), "n_val": len(val)}

    # ======================== A. Predicción de calificaciones =================================
    print("A. Ajuste de hiperparámetros (validación):")
    vu, vi, vy = val["u"].to_numpy(), val["i"].to_numpy(), val["rating"].to_numpy()
    bias_fit = BiasModel().fit(R_fit)
    grid_knn = [{"k": k, "denom_reg": d} for k, d in itertools.product([20, 50, 100], [0.5, 1.0])]
    best_knn = min(
        grid_knn, key=lambda p: rmse(vy, ItemKNN(**p).fit(R_fit, bias_fit).predict(vu, vi))
    )
    grid_als = [{"k": k, "reg": r} for k, r in itertools.product([16, 32, 64], [0.05, 0.1, 0.2])]
    best_als = min(
        grid_als,
        key=lambda p: rmse(vy, ExplicitALS(**p, n_iter=8).fit(R_fit, bias_fit).predict(vu, vi)),
    )
    sgd_probe = FunkSVD(k=32, epochs=30).fit(R_fit, to_csr(val, shape))
    best_epochs = int(np.argmin(sgd_probe.history)) + 1
    print(f"  item-kNN {best_knn} | ALS {best_als} | FunkSVD épocas = {best_epochs}")

    u, i, y = test["u"].to_numpy(), test["i"].to_numpy(), test["rating"].to_numpy()
    bias = BiasModel().fit(R)
    rating_models = {
        "media global": lambda: np.full(len(y), train["rating"].mean()),
        "sesgos (μ + b_u + b_i)": lambda: bias.predict(u, i),
        "item-kNN": lambda: ItemKNN(**best_knn).fit(R, bias).predict(u, i),
        "FunkSVD (SGD)": lambda: FunkSVD(k=32, epochs=best_epochs).fit(R).predict(u, i),
        "ALS explícito": lambda: ExplicitALS(**best_als, n_iter=10).fit(R, bias).predict(u, i),
    }
    rating_results = {}
    for name, fn in rating_models.items():
        t0 = time.perf_counter()
        pred = np.clip(fn(), 1, 5)
        rating_results[name] = {"rmse": rmse(y, pred), "seconds": time.perf_counter() - t0}
    # Misma comparación con partición aleatoria: cuánto más fácil es
    rtr, rte = random_split(ratings, 0.2, seed=SEED)
    Rr = to_csr(rtr, shape)
    rb = BiasModel().fit(Rr)
    random_rmse = {
        "sesgos (μ + b_u + b_i)": rmse(rte["rating"], rb.predict(rte["u"], rte["i"])),
        "ALS explícito": rmse(
            rte["rating"],
            np.clip(
                ExplicitALS(**best_als, n_iter=10)
                .fit(Rr, rb)
                .predict(rte["u"].to_numpy(), rte["i"].to_numpy()),
                1,
                5,
            ),
        ),
    }
    report["rating"] = {
        "best_params": {"item_knn": best_knn, "als": best_als, "funksvd_epochs": best_epochs},
        "temporal": rating_results,
        "aleatoria": random_rmse,
    }
    print("A. RMSE en test (temporal):")
    for name, r in rating_results.items():
        print(f"  {name:<24} {r['rmse']:.4f}  ({r['seconds']:.1f}s)")
    print("  partición aleatoria:", {k: round(v, 4) for k, v in random_rmse.items()})

    # =============================== B. Ranking top-10 =======================================
    print("B. Ajuste de hiperparámetros de ranking (validación, NDCG@10):")
    rel_val = relevant_sets(val)
    pop_fit = popularity_scores(R_fit)

    def val_ndcg(score_fn) -> float:
        return float(
            evaluate_ranking(score_fn, R_fit, rel_val, pop_fit, shape[1])["per_user"]["ndcg"].mean()
        )

    svd_k = max([20, 50, 100, 200], key=lambda k: val_ndcg(PureSVD(k).fit(R_fit).scores))
    ials_alpha = max(
        [0.1, 0.25, 0.5, 1.0, 5.0],
        key=lambda a: val_ndcg(ImplicitALS(k=64, alpha=a, n_iter=10).fit(R_fit).scores),
    )
    knn_rank_k = max(
        [10, 20, 50, 200],
        key=lambda k: val_ndcg(ItemKNN(k=k).fit(R_fit, BiasModel().fit(R_fit)).scores),
    )
    print(f"  PureSVD k = {svd_k} | ALS implícito α = {ials_alpha} | item-kNN k = {knn_rank_k}")

    rel_test = relevant_sets(test)
    funk = FunkSVD(k=32, epochs=best_epochs).fit(R)
    knn_rank = ItemKNN(k=knn_rank_k).fit(R, bias)
    svd = PureSVD(svd_k).fit(R)
    ials = ImplicitALS(k=64, alpha=ials_alpha, n_iter=10).fit(R)
    ranking_models = {
        "aleatorio": lambda us: rng.random((len(us), shape[1])),
        "popularidad": lambda us: np.tile(pop_full, (len(us), 1)),
        "FunkSVD (ordenar por rating predicho)": lambda us: (
            funk.mu + funk.bu[us, None] + funk.bi[None, :] + funk.P[us] @ funk.Q.T
        ),
        "item-kNN": knn_rank.scores,
        "PureSVD": svd.scores,
        "ALS implícito": ials.scores,
    }
    try:
        diversify(np.zeros((1, shape[1])), pop_full)
        ranking_models["ALS implícito + diversify()"] = lambda us: diversify(
            ials.scores(us), pop_full
        )
    except NotImplementedError:
        print("  diversify() sin implementar: se omite esa variante")

    ranking_results, per_user_ndcg = {}, {}
    for name, fn in ranking_models.items():
        ev = evaluate_ranking(fn, R, rel_test, pop_full, shape[1])
        pu = ev["per_user"]
        per_user_ndcg[name] = pu["ndcg"]
        ranking_results[name] = {
            "ndcg@10": float(pu["ndcg"].mean()),
            "ndcg_ic95": bootstrap_ci(pu["ndcg"], rng),
            "recall@10": float(pu["recall"].mean()),
            "precision@10": float(pu["precision"].mean()),
            "hit@10": float(pu["hit"].mean()),
            "cobertura": ev["coverage"],
            "percentil_popularidad": ev["popularity_pct"],
        }
        if name == "ALS implícito":
            best_eval = ev
    users_eval = best_eval["users"]
    report["ranking"] = {
        "n_usuarios": len(users_eval),
        "best_params": {"puresvd_k": svd_k, "ials_alpha": ials_alpha, "knn_k": knn_rank_k},
        "resultados": ranking_results,
    }

    # Por actividad del usuario: ¿a quién sirve el modelo frente a la popularidad?
    activity = np.asarray((R[users_eval] != 0).sum(axis=1)).ravel()
    edges = np.quantile(activity, [0, 0.25, 0.5, 0.75, 1.0])
    buckets = np.clip(np.searchsorted(edges, activity, side="right") - 1, 0, 3)
    by_activity = {}
    for b in range(4):
        m = buckets == b
        by_activity[f"Q{b + 1} ({int(edges[b])}–{int(edges[b + 1])} calif.)"] = {
            name: float(per_user_ndcg[name][m].mean())
            for name in ("popularidad", "item-kNN", "PureSVD", "ALS implícito")
        }
    report["ranking"]["ndcg_por_actividad"] = by_activity

    # Ejemplo cualitativo
    titles = movies.set_index("i")["title"]
    example_user = int(
        users_eval[np.argmin(np.abs(activity - np.median(activity)))]
    )  # actividad mediana
    hist = train[train["u"] == example_user].sort_values("rating", ascending=False).head(5)["i"]
    recs_ex = top_n(ials.scores(np.array([example_user])), R[[example_user]], K)[0]
    report["ejemplo"] = {
        "usuario": example_user,
        "le_gustaron": titles[hist].tolist(),
        "recomendado": titles[recs_ex].tolist(),
        "acierto": [bool(it in rel_test.get(example_user, set())) for it in recs_ex],
    }
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # --- Figuras ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, r in ranking_results.items():
        ax.scatter(r["percentil_popularidad"], r["ndcg@10"], s=60)
        ax.annotate(
            name,
            (r["percentil_popularidad"], r["ndcg@10"]),
            fontsize=7,
            textcoords="offset points",
            xytext=(5, 3),
        )
    ax.set(
        xlabel="percentil de popularidad de lo recomendado (100 = lo más visto)",
        ylabel="NDCG@10",
        title="Precisión vs. sesgo de popularidad",
    )
    save_fig(fig, OUT / "precision_vs_popularidad.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = list(by_activity)
    w = 0.2
    for j, name in enumerate(("popularidad", "item-kNN", "PureSVD", "ALS implícito")):
        ax.bar(np.arange(4) + (j - 1.5) * w, [by_activity[b][name] for b in labels], w, label=name)
    ax.set_xticks(range(4), labels, fontsize=8)
    ax.set(ylabel="NDCG@10", title="Por actividad del usuario en train")
    ax.legend(fontsize=8)
    save_fig(fig, OUT / "por_actividad.png")

    print(f"B. Ranking top-{K} en test ({len(users_eval)} usuarios con al menos un relevante):")
    print(f"  {'modelo':<40}{'NDCG':>7}{'IC 95 %':>18}{'recall':>8}{'cobert.':>9}{'pop. pct':>9}")
    for name, r in ranking_results.items():
        ci = f"[{r['ndcg_ic95'][0]:.3f}, {r['ndcg_ic95'][1]:.3f}]"
        print(
            f"  {name:<40}{r['ndcg@10']:>7.3f}{ci:>18}{r['recall@10']:>8.3f}{r['cobertura']:>9.3f}"
            f"{r['percentil_popularidad']:>9.1f}"
        )
    print(
        "NDCG por actividad:",
        {b: {k: round(v, 3) for k, v in d.items()} for b, d in by_activity.items()},
    )


if __name__ == "__main__":
    main()
