"""Experimentos del integrador S1: libro → temas, temas → libros y libros nuevos, con H1–H3.

Uso:
    uv run python 01-semestre-1/proyecto-integrador-s1/data/download.py
    uv run python 01-semestre-1/proyecto-integrador-s1/run.py [--max-users N]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from applied_ml.metrics import expected_calibration_error
from applied_recsys.factorization import ImplicitALS, PureSVD
from applied_recsys.metrics import (
    catalog_coverage,
    mean_popularity_percentile,
    ranking_metrics,
    top_n,
)
from applied_stats import holm
from book_topics.classify import (
    KNNTopics,
    PrevalenceModel,
    TextTopicModel,
    apply_thresholds,
    book_text,
    fit_thresholds,
)
from book_topics.data import load_books, load_descriptions, load_ratings, load_shelves
from book_topics.evaluation import (
    cluster_bootstrap_diff,
    multilabel_report,
    per_user_diff_bootstrap,
    precision_at_k,
)
from book_topics.openlibrary import agreement, fetch_subjects, subjects_to_topics
from book_topics.recommend import (
    center_ratings,
    hybrid_scores,
    topic_scores,
    topic_scores_with_tiebreak,
    user_topic_profiles,
)
from book_topics.splits import author_split, new_books_holdout, user_split
from book_topics.taxonomy import TAU, label_books, load_taxonomy, topic_matrix
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from common import save_fig, set_seed, setup_style

K = 10
CHUNK = 2000


def _macro(Y, pred):
    return lambda idx: f1_score(Y[idx], pred[idx], average="macro", zero_division=0)


def _eval_ndcg(score_fn, users, exclude, relevant, n_items):
    """NDCG@k por usuario (orden de `users`), en bloques; k = min(10, candidatos − 1)."""
    k = min(K, n_items - 1)
    out = []
    for s in range(0, len(users), CHUNK):
        us = users[s : s + CHUNK]
        recs = top_n(score_fn(us), exclude[us], k)
        out.append((recs, ranking_metrics(recs, [relevant[u] for u in us], k)["ndcg"]))
    recs = np.vstack([r for r, _ in out])
    return np.concatenate([n for _, n in out]), recs


def part_a(books, desc, labels, ratings, cfg, rng):
    keys = list(labels.columns)
    has_text = (
        desc.reindex(books.book_id).fillna("").str.split().str.len().fillna(0).to_numpy()
        >= cfg["min_words"]
    )
    labeled = labels.reindex(books.book_id, fill_value=False).any(axis=1).to_numpy()
    usable = books[has_text & labeled]
    parts = author_split(usable, seed=cfg["seed"])
    texts = book_text(books.set_index("book_id")["title"], desc)
    Y = {k: topic_matrix(labels, v).astype(bool) for k, v in parts.items()}
    X = {k: texts[v].tolist() for k, v in parts.items()}

    def tuned(fit_predict_val, fit_predict_test, grid):
        best = max(
            grid,
            key=lambda g: f1_score(
                Y["val"],
                apply_thresholds(*_thr(fit_predict_val(g))),
                average="macro",
                zero_division=0,
            ),
        )
        P_val, P_test = fit_predict_val(best), fit_predict_test(best)
        thr = fit_thresholds(Y["val"], P_val)
        return best, P_test, apply_thresholds(P_test, thr)

    def _thr(P_val):
        return P_val, fit_thresholds(Y["val"], P_val)

    models = {}
    text_models = {}

    def logit(C):
        if C not in text_models:
            text_models[C] = TextTopicModel(
                C=C, min_df=cfg["min_df"], calibration_cv=cfg["cv"]
            ).fit(X["train"], Y["train"])
        return text_models[C]

    C, P_lr, pred_lr = tuned(
        lambda c: logit(c).predict_proba(X["val"]),
        lambda c: logit(c).predict_proba(X["test"]),
        cfg["C"],
    )
    models["logistica"] = (P_lr, pred_lr, {"C": C})
    vec = logit(C)
    Xtr, Xva, Xte = vec.transform(X["train"]), vec.transform(X["val"]), vec.transform(X["test"])
    k, P_knn, pred_knn = tuned(
        lambda kk: KNNTopics(kk).fit(Xtr, Y["train"]).predict_proba(Xva),
        lambda kk: KNNTopics(kk).fit(Xtr, Y["train"]).predict_proba(Xte),
        cfg["k"],
    )
    models["knn"] = (P_knn, pred_knn, {"k": k})
    prev = PrevalenceModel().fit(Y["train"])
    P_prev = prev.predict_proba(len(parts["test"]))
    models["prevalencia"] = (
        P_prev,
        apply_thresholds(P_prev, fit_thresholds(Y["val"], prev.predict_proba(len(parts["val"])))),
        {},
    )

    # Lectores → temas: factores de ALS implícito por libro (sin usar etiquetas)
    n_users, n_books = int(ratings.user_id.max()), int(books.book_id.max())
    R = sparse.csr_matrix(
        (ratings.rating.to_numpy(float), (ratings.user_id - 1, ratings.book_id - 1)),
        shape=(n_users, n_books),
    )
    als = ImplicitALS(k=cfg["factors"], alpha=1.0, n_iter=cfg["als_iter"]).fit(R)
    F = {kk: als.Y[v - 1] for kk, v in parts.items()}
    P_val_r = np.column_stack(
        [
            LogisticRegression(max_iter=2000)
            .fit(F["train"], Y["train"][:, t])
            .predict_proba(F["val"])[:, 1]
            if Y["train"][:, t].any() and not Y["train"][:, t].all()
            else np.full(len(F["val"]), Y["train"][:, t].mean())
            for t in range(len(keys))
        ]
    )
    P_te_r = np.column_stack(
        [
            LogisticRegression(max_iter=2000)
            .fit(F["train"], Y["train"][:, t])
            .predict_proba(F["test"])[:, 1]
            if Y["train"][:, t].any() and not Y["train"][:, t].all()
            else np.full(len(F["test"]), Y["train"][:, t].mean())
            for t in range(len(keys))
        ]
    )
    models["lectores"] = (P_te_r, apply_thresholds(P_te_r, fit_thresholds(Y["val"], P_val_r)), {})

    groups = usable.set_index("book_id").loc[parts["test"], "main_author"].to_numpy()
    results = {}
    for name, (P, pred, params) in models.items():
        rep = multilabel_report(Y["test"], pred)
        results[name] = {
            "micro_f1": rep["micro_f1"],
            "macro_f1": rep["macro_f1"],
            "precision_at_3": precision_at_k(Y["test"], P, 3),
            "ece": expected_calibration_error(Y["test"].ravel(), P.ravel()),
            "per_topic_f1": dict(zip(keys, rep["per_topic_f1"].round(4).tolist(), strict=True)),
            "params": params,
        }
    h1 = cluster_bootstrap_diff(
        _macro(Y["test"], models["logistica"][1]),
        _macro(Y["test"], models["knn"][1]),
        groups,
        cfg["n_boot"],
        cfg["seed"],
    )
    exploratory = cluster_bootstrap_diff(
        _macro(Y["test"], models["logistica"][1]),
        _macro(Y["test"], models["lectores"][1]),
        groups,
        cfg["n_boot"],
        cfg["seed"],
    )
    # Análisis de errores
    prevalence = Y["train"].mean(axis=0)
    f1_lr = np.array(list(results["logistica"]["per_topic_f1"].values()))
    rho = spearmanr(prevalence, f1_lr)[0] if len(keys) > 2 else float("nan")
    fp = pred_lr & ~Y["test"]
    confusions = sorted(
        (
            (int((fp[:, a] & Y["test"][:, b]).sum()), keys[a], keys[b])
            for a in range(len(keys))
            for b in range(len(keys))
            if a != b
        ),
        reverse=True,
    )[:5]
    words = desc.reindex(parts["test"]).fillna("").str.split().str.len().to_numpy()
    quart = pd.qcut(words, 4, labels=False, duplicates="drop")
    by_length = {
        int(q): float(
            f1_score(Y["test"][quart == q], pred_lr[quart == q], average="micro", zero_division=0)
        )
        for q in np.unique(quart)
    }
    return (
        {
            "n_libros": {k: len(v) for k, v in parts.items()},
            "modelos": results,
            "spearman_prevalencia_f1": float(rho),
            "confusiones": confusions,
            "f1_por_cuartil_largo": by_length,
            "exploratoria_texto_vs_lectores": exploratory,
        },
        h1,
        logit(C),
    )


def part_b_and_new(books, desc, labels, ratings, text_model_factory, cfg, rng, max_users):
    n_books = int(books.book_id.max())
    new = new_books_holdout(books.book_id.to_numpy(), 0.1, seed=cfg["seed"])
    new_idx = np.sort(new - 1)
    r = ratings.assign(u=ratings.user_id - 1, i=ratings.book_id - 1)
    old = r[~r.book_id.isin(new)]
    train, val, test = user_split(old)
    n_users = int(r.u.max()) + 1
    T = topic_matrix(labels, books.book_id.to_numpy())
    T_old = T.copy()
    T_old[new_idx] = 0
    R_tr = sparse.csr_matrix(
        (train.rating.to_numpy(float), (train.u, train.i)), shape=(n_users, n_books)
    )
    R_seen = sparse.csr_matrix(
        (np.ones(len(train) + len(val)), (np.r_[train.u, val.u], np.r_[train.i, val.i])),
        shape=(n_users, n_books),
    )
    profiles = user_topic_profiles(center_ratings(train, n_users, n_books), T_old)
    popularity = np.asarray((R_tr != 0).sum(axis=0)).ravel().astype(float)

    def rel(df):
        pos = df[df.rating >= 4]
        return {int(u): set(g.i) for u, g in pos.groupby("u")}

    rel_val, rel_test = rel(val), rel(test)
    users_val = np.array(sorted(rel_val))
    users_test = np.array(sorted(rel_test))
    if max_users and len(users_test) > max_users:
        users_test = np.sort(rng.choice(users_test, max_users, replace=False))
        users_val = np.sort(rng.choice(users_val, min(max_users, len(users_val)), replace=False))

    def collab_candidates():
        for a in cfg["ials_alpha"]:
            yield (
                f"ALS implícito (α={a})",
                ImplicitALS(k=cfg["factors"], alpha=a, n_iter=cfg["als_iter"]).fit(R_tr).scores,
            )
        for k in cfg["svd_k"]:
            yield f"PureSVD (k={k})", PureSVD(min(k, min(R_tr.shape) - 1)).fit(R_tr).scores

    best_name, best_fn, best_val = None, None, -1.0
    for name, fn in collab_candidates():
        v = _eval_ndcg(fn, users_val, R_tr, rel_val, n_books)[0].mean()
        if v > best_val:
            best_name, best_fn, best_val = name, fn, v

    def topic_fn(us):
        return topic_scores_with_tiebreak(profiles[us], T_old, popularity)

    best_alpha = max(
        cfg["alphas"],
        key=lambda a: _eval_ndcg(
            lambda us: hybrid_scores(best_fn(us), topic_fn(us), a),
            users_val,
            R_tr,
            rel_val,
            n_books,
        )[0].mean(),
    )
    scorers = {
        "popularidad": lambda us: np.tile(popularity, (len(us), 1)),
        "colaborativo": best_fn,
        "temas": topic_fn,
        "hibrido": lambda us: hybrid_scores(best_fn(us), topic_fn(us), best_alpha),
    }
    per_user, summary = {}, {}
    for name, fn in scorers.items():
        nd, recs = _eval_ndcg(fn, users_test, R_seen, rel_test, n_books)
        per_user[name] = nd
        summary[name] = {
            "ndcg@10": float(nd.mean()),
            "cobertura": catalog_coverage(recs, n_books),
            "percentil_popularidad": mean_popularity_percentile(recs, popularity),
        }
    history = np.asarray((R_tr[users_test] != 0).sum(axis=1)).ravel()
    q = np.quantile(history, [0.25, 0.5, 0.75])
    bucket = np.searchsorted(q, history, side="right")
    by_history = {
        f"Q{b + 1}": {n: float(per_user[n][bucket == b].mean()) for n in scorers}
        for b in range(4)
        if (bucket == b).any()
    }
    q1 = bucket == 0
    h2 = per_user_diff_bootstrap(
        per_user["hibrido"][q1], per_user["colaborativo"][q1], cfg["n_boot"], cfg["seed"]
    )

    # Libros nuevos: temas predichos desde la descripción con un modelo que no los vio
    has_text = desc.reindex(books.book_id).fillna("").str.len().to_numpy() > 0
    known = books[(~books.book_id.isin(new)).to_numpy() & has_text]
    known = known[labels.reindex(known.book_id, fill_value=False).any(axis=1).to_numpy()]
    texts = book_text(books.set_index("book_id")["title"], desc)
    model = text_model_factory().fit(
        texts[known.book_id].tolist(), topic_matrix(labels, known.book_id.to_numpy()).astype(bool)
    )
    new_with_text = np.array([b for b in np.sort(new) if desc.get(b, "")])
    T_new = model.predict_proba(texts[new_with_text].tolist())
    new_cols = new_with_text - 1
    rel_new = {
        int(u): set(g.i) for u, g in r[r.book_id.isin(new_with_text) & (r.rating >= 4)].groupby("u")
    }
    users_new = np.array(sorted(u for u in rel_new if profiles[u].any()))
    if max_users and len(users_new) > max_users:
        users_new = np.sort(rng.choice(users_new, max_users, replace=False))
    col_pos = {c: j for j, c in enumerate(new_cols)}
    rel_new_local = {u: {col_pos[i] for i in rel_new[u]} for u in users_new}
    pop_new = (
        r[r.book_id.isin(new_with_text)]
        .groupby("i")
        .size()
        .reindex(new_cols, fill_value=0)
        .to_numpy(float)
    )
    empty = sparse.csr_matrix((n_users, len(new_cols)))
    new_scorers = {
        "temas_predichos": lambda us: topic_scores(profiles[us], T_new),
        "popularidad (optimista)": lambda us: np.tile(pop_new, (len(us), 1)),
        "aleatorio": lambda us: rng.random((len(us), len(new_cols))),
    }
    nd_new = {
        n: _eval_ndcg(fn, users_new, empty, rel_new_local, len(new_cols))[0]
        for n, fn in new_scorers.items()
    }
    h3 = per_user_diff_bootstrap(
        nd_new["temas_predichos"], nd_new["popularidad (optimista)"], cfg["n_boot"], cfg["seed"]
    )
    return (
        {
            "colaborativo_elegido": best_name,
            "alpha_hibrido": best_alpha,
            "n_usuarios": len(users_test),
            "modelos": summary,
            "por_historial": by_history,
        },
        {
            "n_libros_nuevos": len(new_cols),
            "n_usuarios": len(users_new),
            "ndcg@10": {n: float(v.mean()) for n, v in nd_new.items()},
        },
        h2,
        h3,
        per_user,
        by_history,
    )


def validate_labels(books, labels, data_dir, rng, n=300):
    try:
        sample = books[books.isbn.str.len() >= 9].sample(
            n=min(n, len(books)), random_state=int(rng.integers(1e9))
        )
        subjects = fetch_subjects(sample.isbn.tolist(), Path(data_dir) / "openlibrary_cache.json")
        ol = {
            int(b): subjects_to_topics(subjects[i])
            for b, i in zip(sample.book_id, sample.isbn, strict=True)
            if i in subjects
        }
        return {
            "n_libros": len(ol),
            "kappa_por_tema": agreement(labels, ol).to_dict(orient="records"),
        }
    except Exception as e:  # la validación no bloquea el resto (spec 10)
        return f"no realizada: {e}"


def run(
    data_dir: Path,
    out_dir: Path,
    taxonomy_path: Path,
    *,
    tau: float = TAU,
    max_users: int | None = None,
    smoke: bool = False,
    seed: int = 42,
) -> dict:
    rng = set_seed(seed)
    setup_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "seed": seed,
        "n_boot": 50 if smoke else 2000,
        "C": [4.0] if smoke else [1.0, 4.0, 16.0],
        "k": [5] if smoke else [10, 20, 50],
        "min_df": 1 if smoke else 2,
        "cv": 2 if smoke else 3,
        "factors": 8 if smoke else 64,
        "als_iter": 3 if smoke else 10,
        "ials_alpha": [1.0] if smoke else [0.25, 1.0],
        "svd_k": [3] if smoke else [50, 100],
        "alphas": [0.0, 0.5, 1.0] if smoke else [0.0, 0.25, 0.5, 0.75, 1.0],
        "min_words": 1 if smoke else 20,
    }
    books, desc, ratings = load_books(data_dir), load_descriptions(data_dir), load_ratings(data_dir)
    labels = label_books(load_shelves(data_dir), load_taxonomy(taxonomy_path), tau)
    a, h1, _ = part_a(books, desc, labels, ratings, cfg, rng)

    def factory():
        return TextTopicModel(
            C=a["modelos"]["logistica"]["params"]["C"],
            min_df=cfg["min_df"],
            calibration_cv=cfg["cv"],
        )

    b, new, h2, h3, _, by_history = part_b_and_new(
        books, desc, labels, ratings, factory, cfg, rng, max_users
    )
    p_adj = holm([h1["p_value"], h2["p_value"], h3["p_value"]])
    hyp = {}
    for (name, h), p in zip((("H1", h1), ("H2", h2), ("H3", h3)), p_adj, strict=True):
        rejects = bool(p < 0.05)
        # “rechaza H0” es bilateral: apoyar la hipótesis exige además la dirección prerregistrada
        hyp[name] = {
            **h,
            "p_holm": float(p),
            "rechaza_h0": rejects,
            "apoya_hipotesis": rejects and h["diff"] > 0,
        }
    report = {
        "parte_a": a,
        "parte_b": b,
        "libros_nuevos": new,
        "hipotesis": hyp,
        "validacion_etiquetas": "omitida (smoke)"
        if smoke
        else validate_labels(books, labels, data_dir, rng),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str)
    )

    keys = list(labels.columns)
    f1 = a["modelos"]["logistica"]["per_topic_f1"]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.25 * len(keys))))
    order = sorted(f1, key=f1.get)
    ax.barh(order, [f1[k] for k in order])
    ax.set(xlabel="F1 en test", title="Libro → temas: F1 por tema (logística)")
    save_fig(fig, out_dir / "f1_por_tema.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(next(iter(by_history.values())))
    for j, n in enumerate(names):
        ax.bar(
            np.arange(len(by_history)) + (j - 1.5) * 0.2,
            [by_history[q][n] for q in by_history],
            0.2,
            label=n,
        )
    ax.set_xticks(range(len(by_history)), list(by_history))
    ax.set(ylabel="NDCG@10", title="Recomendación por cuartil de historial")
    ax.legend(fontsize=8)
    save_fig(fig, out_dir / "ndcg_por_historial.png")
    plt.close("all")
    return report


def main() -> None:
    from book_topics import MODULE

    from common import data_dir

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-users", type=int, default=None)
    args = parser.parse_args()
    d = data_dir(MODULE)
    report = run(
        d,
        Path(__file__).resolve().parent / "resultados",
        d.parent / "temas.yaml",
        max_users=args.max_users,
    )
    for name, h in report["hipotesis"].items():
        print(
            f"{name}: Δ = {h['diff']:+.4f} IC [{h['ci'][0]:+.4f}, {h['ci'][1]:+.4f}]"
            f" p = {h['p_value']:.4f} p_Holm = {h['p_holm']:.4f}"
            f" → {'rechaza' if h['rechaza_h0'] else 'no rechaza'} H0;"
            f" {'apoya' if h['apoya_hipotesis'] else 'NO apoya'} la hipótesis"
        )


if __name__ == "__main__":
    main()
