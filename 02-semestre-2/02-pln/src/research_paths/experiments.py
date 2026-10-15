"""Experimentos del proyecto: capa 1 (AL-CPL), capa 2 (OpenAlex) e hipótesis → resultados/.

Todo entra por `Context`, así el smoke corre con Wikipedia/OpenAlex falsos, un codificador de
hashing y modelos diminutos, y `proyecto/run.py` con los reales.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import save_fig
from research_paths.alcpl import DOMAINS, folds, load_alcpl, map_concepts, split_fold
from research_paths.concepts import concept_text
from research_paths.evaluation import (
    average_precision,
    best_threshold,
    cluster_bootstrap,
    f1_binary,
    holm,
)
from research_paths.prereq.features import fit_pair_logreg, pair_features, select_c
from research_paths.prereq.refd import refd_scores
from research_paths.readings import (
    EVAL_FIELDS,
    ORDERS,
    intro_training_set,
    ranking_orders,
    sample_eval_topics,
    topic_candidates,
    topic_metrics,
)
from research_paths.retrieval import Encoder
from research_paths.textclf import (
    TrainConfig,
    predict_proba,
    save_classifier,
    train_classifier,
)


@dataclass
class Context:
    data_dir: Path
    out_dir: Path
    models_dir: Path
    wiki_en: object
    wiki_es: object
    oa: object
    encoder: Encoder
    make_model: Callable[[list[str]], tuple]  # textos de entrenamiento → (modelo, tokenizador)
    train_cfg: TrainConfig
    judge: object | None = None
    device: str = "cpu"
    seed: int = 42
    n_boot: int = 2000
    per_field: int = 15
    intro_topics_per_field: int = 25
    fields: tuple[str, ...] = EVAL_FIELDS
    log: Callable[..., None] = field(default=print)


# ---------------------------------------------------------------- capa 1 · prerrequisitos


def load_concepts(ctx: Context) -> dict:
    df = load_alcpl(ctx.data_dir / "alcpl")
    mapping = ctx.wiki_en.langlinks(sorted(set(df.a) | set(df.b)), "es")
    pairs, stats = map_concepts(df, mapping)
    es = sorted(set(pairs.a_es) | set(pairs.b_es))
    en = sorted(set(pairs.a) | set(pairs.b))
    ext_es, ext_en = ctx.wiki_es.extracts(es), ctx.wiki_en.extracts(en)
    stats |= {
        "concepts_es": len(es),
        "es_without_extract": sum(not ext_es[c] for c in es),
        "en_without_extract": sum(not ext_en[c] for c in en),
    }
    ctx.log(f"AL-CPL → ES: {stats}")
    return {
        "pairs": pairs,
        "stats": stats,
        "text_es": {c: concept_text(c, ext_es[c]) for c in es},
        "text_en": {c: concept_text(c, ext_en[c]) for c in en},
        "links_es": {c: set(ctx.wiki_es.links(c)) for c in es},
    }


def _texts(df: pd.DataFrame, texts: dict, col_a: str, col_b: str) -> tuple[list[str], list[str]]:
    return [texts[x] for x in df[col_a]], [texts[x] for x in df[col_b]]


def _cross_encoder(ctx, train, val, test, texts, col_a, col_b) -> tuple[np.ndarray, np.ndarray]:
    tr_a, tr_b = _texts(train, texts, col_a, col_b)
    va_a, va_b = _texts(val, texts, col_a, col_b)
    te_a, te_b = _texts(test, texts, col_a, col_b)
    model, tok = ctx.make_model(tr_a + tr_b)
    model, tok, _ = train_classifier(
        tr_a,
        train.label.to_numpy(),
        va_a,
        val.label.to_numpy(),
        ctx.train_cfg,
        texts_b=tr_b,
        val_b=va_b,
        model=model,
        tokenizer=tok,
        device=ctx.device,
        log=ctx.log,
    )
    kw = {"max_length": ctx.train_cfg.max_length, "device": ctx.device}
    return predict_proba(model, tok, va_a, va_b, **kw), predict_proba(model, tok, te_a, te_b, **kw)


def run_layer1(ctx: Context, data: dict) -> tuple[pd.DataFrame, dict]:
    pairs, text_es, text_en = data["pairs"], data["text_es"], data["text_en"]
    concepts = set(text_es)
    names = sorted(text_es)
    emb = dict(zip(names, ctx.encoder([text_es[c] for c in names], "query"), strict=True))

    def features(d):
        return pair_features(np.stack([emb[a] for a in d.a_es]), np.stack([emb[b] for b in d.b_es]))

    rows = []
    for fold in folds():
        train, val, test = split_fold(pairs, fold)
        ctx.log(f"Ronda test={fold.test}: {len(train)}/{len(val)}/{len(test)} pares")
        out = test[["domain", "a", "b", "a_es", "b_es", "label"]].copy()
        y_val = val.label.to_numpy()

        def put(name, s_val, s_test, out=out, y_val=y_val):
            out[f"score_{name}"] = s_test
            out[f"pred_{name}"] = s_test >= best_threshold(y_val, s_val)

        prev = float(train.label.mean())
        put("prevalencia", np.full(len(val), prev), np.full(len(test), prev))
        links = data["links_es"]
        put("refd", refd_scores(val, links, concepts), refd_scores(test, links, concepts))
        C = select_c(features(train), train.label, features(val), y_val)
        lr = fit_pair_logreg(features(train), train.label, C)
        put(
            "embeddings",
            lr.predict_proba(features(val))[:, 1],
            lr.predict_proba(features(test))[:, 1],
        )
        put("cross_es", *_cross_encoder(ctx, train, val, test, text_es, "a_es", "b_es"))
        put("cross_en", *_cross_encoder(ctx, train, val, test, text_en, "a", "b"))
        if ctx.judge is not None:
            items = [
                (a, text_es[a], b, text_es[b]) for a, b in zip(test.a_es, test.b_es, strict=True)
            ]
            answers = ctx.judge.judge(items)
            out["pred_llm"] = [a is True for a in answers]
            out["score_llm"] = out.pred_llm.astype(float)
            out["llm_unparsed"] = [a is None for a in answers]
        rows.append(out)
    preds = pd.concat(rows, ignore_index=True)
    models = ["prevalencia", "refd", "embeddings", "cross_es", "cross_en"]
    if "pred_llm" in preds:
        models.append("llm")
    metrics = {"modelos": {m: _layer1_metrics(preds, m) for m in models}}
    if "llm_unparsed" in preds:
        metrics["llm_no_interpretables"] = int(preds.llm_unparsed.sum())
    return preds, metrics


def _layer1_metrics(preds: pd.DataFrame, model: str) -> dict:
    def one(d):
        y = d.label.to_numpy()
        return {
            "f1": f1_binary(y, d[f"pred_{model}"]),
            "ap": average_precision(y, d[f"score_{model}"]),
            "n": len(d),
            "positivos": int(y.sum()),
        }

    return {**one(preds), "por_dominio": {d: one(g) for d, g in preds.groupby("domain")}}


def train_final_prereq(ctx: Context, data: dict) -> None:
    """Cross-encoder ES para la CLI: entrena con 3 dominios y valida en el más pequeño."""
    pairs = data["pairs"]
    val_domain = pairs.domain.value_counts().idxmin()
    val = pairs[pairs.domain == val_domain]
    val_c = set(val.a_es) | set(val.b_es)
    train = pairs[(pairs.domain != val_domain) & ~pairs.a_es.isin(val_c) & ~pairs.b_es.isin(val_c)]
    tr_a, tr_b = _texts(train, data["text_es"], "a_es", "b_es")
    va_a, va_b = _texts(val, data["text_es"], "a_es", "b_es")
    model, tok = ctx.make_model(tr_a + tr_b)
    model, tok, _ = train_classifier(
        tr_a,
        train.label.to_numpy(),
        va_a,
        val.label.to_numpy(),
        ctx.train_cfg,
        texts_b=tr_b,
        val_b=va_b,
        model=model,
        tokenizer=tok,
        device=ctx.device,
        log=ctx.log,
    )
    save_classifier(model, tok, ctx.models_dir / "prereq-es")


# ---------------------------------------------------------------- capa 2 · lecturas


def run_layer2(ctx: Context) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    topics = sample_eval_topics(ctx.oa, fields=ctx.fields, per_field=ctx.per_field, seed=ctx.seed)
    ctx.log(f"Temas evaluados: {len(topics)}")
    intro = intro_training_set(
        ctx.oa,
        topics,
        fields=ctx.fields,
        topics_per_field=ctx.intro_topics_per_field,
        seed=ctx.seed + 1,
    )
    rng = np.random.default_rng(ctx.seed)
    tids = np.array(sorted(intro.topic_id.unique()))
    rng.shuffle(tids)
    val_topics = set(tids[: max(1, len(tids) // 5)])
    tr, va = intro[~intro.topic_id.isin(val_topics)], intro[intro.topic_id.isin(val_topics)]
    model, tok = ctx.make_model(tr.text.tolist())
    model, tok, history = train_classifier(
        tr.text.tolist(),
        tr.label.to_numpy(),
        va.text.tolist(),
        va.label.to_numpy(),
        ctx.train_cfg,
        model=model,
        tokenizer=tok,
        device=ctx.device,
        log=ctx.log,
    )
    save_classifier(model, tok, ctx.models_dir / "intro")
    kw = {"max_length": ctx.train_cfg.max_length, "device": ctx.device}
    rows, pools = [], []
    for t in topics:
        cands = topic_candidates(ctx.oa, t)
        p = predict_proba(model, tok, cands.text.tolist(), **kw) if len(cands) else np.zeros(0)
        m = topic_metrics(cands, t.gold, ranking_orders(cands, t.name, ctx.encoder, p))
        row = {
            "topic_id": t.topic_id, "name": t.name, "field": t.field, "review_id": t.review_id,
            "review_year": t.review_year, "n_gold": len(t.gold), "n_candidates": m["n_candidates"],
            "ceiling": m["ceiling"], "abstract_frac": m["abstract_frac"],
        }  # fmt: skip
        for name, v in m["orders"].items():
            row[f"recall_{name}"], row[f"ndcg_{name}"] = v["recall"], v["ndcg"]
        rows.append(row)
        pools.append(cands.assign(p_intro=p, topic_id=t.topic_id))
    topics_df = pd.DataFrame(rows)
    pool = pd.concat(pools, ignore_index=True) if pools else pd.DataFrame()
    pool = pool[pool.abstract != ""] if len(pool) else pool
    sample = pool.sample(min(100, len(pool)), random_state=ctx.seed) if len(pool) else pool
    sample = sample[["id", "topic_id", "title", "abstract", "p_intro"]] if len(sample) else sample
    metrics = {
        "n_temas": len(topics_df),
        "temas_por_campo": {k: int(v) for k, v in topics_df.field.value_counts().items()},
        "techo_medio": float(topics_df.ceiling.mean()) if len(topics_df) else 0.0,
        "fraccion_con_resumen": float(topics_df.abstract_frac.mean()) if len(topics_df) else 0.0,
        "intro": {
            "n_train": len(tr),
            "n_val": len(va),
            "ap_val": max(h["val_ap"] for h in history),
            "historia": history,
        },
        "ordenes": {
            name: {
                "recall@20": float(topics_df[f"recall_{name}"].mean()),
                "ndcg@20": float(topics_df[f"ndcg_{name}"].mean()),
            }
            for name in ORDERS
        },
    }
    return topics_df, metrics, sample


def intro_agreement(sample: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Acuerdo del clasificador introductorio con las etiquetas «¿sirve para empezar?»."""
    m = sample.merge(labels[["id", "sirve"]], on="id")
    y, p = m.sirve.to_numpy(), m.p_intro.to_numpy()
    return {
        "n": len(m),
        "positivos": int(y.sum()),
        "ap": average_precision(y, p),
        "f1@0,5": f1_binary(y, p >= 0.5),
    }


# ---------------------------------------------------------------- hipótesis


def hypotheses(preds: pd.DataFrame, topics_df: pd.DataFrame, ctx: Context) -> dict:
    rng = np.random.default_rng(ctx.seed)
    y = preds.label.to_numpy()
    concepts = preds.a_es.to_numpy()

    def f1_diff(m1, m2):
        p1, p2 = preds[f"pred_{m1}"].to_numpy(), preds[f"pred_{m2}"].to_numpy()
        return lambda idx: f1_binary(y[idx], p1[idx]) - f1_binary(y[idx], p2[idx])

    H = {
        "H1": {"descripcion": "F1(cross-encoder ES) − F1(RefD)",
               **cluster_bootstrap(concepts, f1_diff("cross_es", "refd"), rng, ctx.n_boot)},
        "H2": {"descripcion": "F1(juez LLM) − F1(cross-encoder ES)", "evaluada": False},
        "H3": {"descripcion": "F1(cross-encoder EN) − F1(cross-encoder ES)",
               **cluster_bootstrap(concepts, f1_diff("cross_en", "cross_es"), rng, ctx.n_boot)},
    }  # fmt: skip
    if "pred_llm" in preds:
        H["H2"] = {
            "descripcion": H["H2"]["descripcion"],
            **cluster_bootstrap(concepts, f1_diff("llm", "cross_es"), rng, ctx.n_boot),
        }
    diff = (topics_df.recall_combinado - topics_df.recall_citas).to_numpy()
    H["H4"] = {
        "descripcion": "recall@20(combinado) − recall@20(más citados)",
        **cluster_bootstrap(
            topics_df.topic_id.to_numpy(), lambda i: float(diff[i].mean()), rng, ctx.n_boot
        ),
    }
    evaluated = [k for k in ("H1", "H2", "H3", "H4") if H[k].get("evaluada", True)]
    for k, p_holm in zip(evaluated, holm([H[k]["p"] for k in evaluated]), strict=True):
        H[k].pop("_boots")
        H[k]["p_holm"] = p_holm
        H[k]["se_apoya"] = bool(p_holm < 0.05 and H[k]["delta"] > 0)
    return H


# ---------------------------------------------------------------- figuras y orquestación


def _plot_layer1(metrics: dict, path: Path) -> None:
    models = list(metrics["modelos"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(models)
    for k, m in enumerate(models):
        vals = [metrics["modelos"][m]["por_dominio"].get(d, {}).get("f1", 0.0) for d in DOMAINS]
        ax.bar(np.arange(len(DOMAINS)) + k * width, vals, width, label=m)
    ax.set_xticks(np.arange(len(DOMAINS)) + 0.4 - width / 2, DOMAINS)
    ax.set_ylabel("F1 (dominio no visto)")
    ax.set_title("Prerrequisitos: F1 por dominio de test")
    ax.legend(ncols=3, fontsize=8)
    save_fig(fig, path)
    plt.close(fig)


def _plot_layer2(metrics: dict, path: Path) -> None:
    names = list(metrics["ordenes"])
    fig, ax = plt.subplots()
    ax.bar(names, [metrics["ordenes"][n]["recall@20"] for n in names])
    ax.axhline(metrics["techo_medio"], ls="--", c="gray", label="techo (candidatos)")
    ax.set_ylabel("recall@20 medio")
    ax.set_title("Lecturas: referencias de la revisión recuperadas")
    ax.legend()
    save_fig(fig, path)
    plt.close(fig)


def _api_stats(ctx: Context) -> dict:
    out = {}
    for name in ("wiki_en", "wiki_es", "oa"):
        stats = getattr(getattr(getattr(ctx, name), "client", None), "stats", None)
        if stats is not None:
            out[name] = dict(stats)
    judge_stats = getattr(ctx.judge, "stats", None)
    if judge_stats is not None:
        out["llm"] = dict(judge_stats)
    return out


def run_all(ctx: Context) -> dict:
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    data = load_concepts(ctx)
    preds, m1 = run_layer1(ctx, data)
    train_final_prereq(ctx, data)
    topics_df, m2, sample = run_layer2(ctx)
    metrics = {
        "capa1": {"cobertura": data["stats"], **m1},
        "capa2": m2,
        "hipotesis": hypotheses(preds, topics_df, ctx),
        "api": _api_stats(ctx),
    }
    text = json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False)
    (ctx.out_dir / "metrics.json").write_text(text + "\n")
    preds.to_csv(ctx.out_dir / "capa1_predicciones.csv", index=False)
    topics_df.to_csv(ctx.out_dir / "capa2_temas.csv", index=False)
    sample.to_csv(ctx.out_dir / "intro_muestra.csv", index=False)
    _plot_layer1(m1, ctx.out_dir / "f1_por_dominio.png")
    _plot_layer2(m2, ctx.out_dir / "recall_por_orden.png")
    return json.loads(text)
