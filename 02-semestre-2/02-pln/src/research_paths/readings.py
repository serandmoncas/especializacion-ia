"""Lecturas: temas de evaluación en OpenAlex, candidatos y orden «empieza aquí».

La capa 2 se evalúa sobre temas de investigación de OpenAlex: en cada tema, las referencias de su
revisión más citada son la lista de referencia, y los candidatos son trabajos del mismo tema
publicados antes de esa revisión (sin «ver el futuro»).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research_paths.evaluation import ndcg_at_k, recall_at_k
from research_paths.openalex import abstract_text, short_id
from research_paths.retrieval import BM25, Encoder, cosine_scores, simple_tokenize

# Computer Science, Mathematics, Physics and Astronomy, Agricultural and Biological Sciences.
EVAL_FIELDS = ("17", "26", "31", "11")
REVIEW_FILTER = "type:review,publication_year:2010-2024,referenced_works_count:>19"
ORDERS = ("openalex", "citas", "bm25", "denso", "combinado")
COLUMNS = [
    "id",
    "title",
    "abstract",
    "text",
    "year",
    "citations",
    "type",
    "language",
    "relevance_rank",
]


@dataclass(frozen=True)
class EvalTopic:
    topic_id: str
    name: str
    field: str
    review_id: str
    review_year: int
    gold: frozenset[str]


def works_frame(works: list[dict]) -> pd.DataFrame:
    """Trabajos de OpenAlex → columnas fijas; los nulos pasan a "" o 0."""
    rows = []
    for rank, w in enumerate(works):
        title = (w.get("title") or "").strip()
        abstract = abstract_text(w.get("abstract_inverted_index"))
        rows.append(
            {
                "id": short_id(w.get("id") or ""),
                "title": title,
                "abstract": abstract,
                "text": f"{title}. {abstract}" if abstract else title,
                "year": int(w.get("publication_year") or 0),
                "citations": int(w.get("cited_by_count") or 0),
                "type": w.get("type") or "",
                "language": w.get("language") or "",
                "relevance_rank": rank,
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def _shuffled_topics(oa, field: str, rng: np.random.Generator) -> list[dict]:
    topics = sorted(oa.topics(filter=f"field.id:{field}"), key=lambda t: t["id"])
    return [topics[i] for i in rng.permutation(len(topics))]


def sample_eval_topics(
    oa, fields=EVAL_FIELDS, per_field: int = 15, seed: int = 42
) -> list[EvalTopic]:
    """Por campo, los primeros `per_field` temas (barajados con semilla) con una revisión válida."""
    rng = np.random.default_rng(seed)
    out = []
    for field in fields:
        found = 0
        for topic in _shuffled_topics(oa, field, rng):
            if found == per_field:
                break
            tid = short_id(topic["id"])
            reviews = oa.works(
                filter=f"primary_topic.id:{tid},{REVIEW_FILTER}",
                sort="cited_by_count:desc",
                max_results=1,
                select="id,publication_year,referenced_works",
            )
            if not reviews:
                continue
            review = reviews[0]
            gold = frozenset(short_id(x) for x in review.get("referenced_works") or [])
            if not gold or not review.get("publication_year"):
                continue
            name = topic.get("display_name") or tid
            year = int(review["publication_year"])
            out.append(EvalTopic(tid, name, field, short_id(review["id"]), year, gold))
            found += 1
    return out


def topic_candidates(oa, topic: EvalTopic, n: int = 200) -> pd.DataFrame:
    works = oa.works(
        filter=f"primary_topic.id:{topic.topic_id},publication_year:<{topic.review_year}",
        search=topic.name,
        max_results=n,
    )
    df = works_frame(works)
    df = df[df.id != topic.review_id].reset_index(drop=True)
    df["relevance_rank"] = np.arange(len(df))
    return df


def intro_training_set(
    oa,
    eval_topics: list[EvalTopic],
    fields=EVAL_FIELDS,
    topics_per_field: int = 25,
    per_topic: int = 20,
    seed: int = 43,
) -> pd.DataFrame:
    """Etiqueta débil: `review` = 1, `article` = 0, balanceada por tema, de temas NO evaluados y
    sin las revisiones de referencia ni sus referencias."""
    rng = np.random.default_rng(seed)
    skip_topics = {t.topic_id for t in eval_topics}
    skip_works = set().union(*(t.gold | {t.review_id} for t in eval_topics))
    frames = []
    for field in fields:
        used = 0
        for topic in _shuffled_topics(oa, field, rng):
            if used == topics_per_field:
                break
            tid = short_id(topic["id"])
            if tid in skip_topics:
                continue
            by_kind = {}
            for kind in ("review", "article"):
                w = works_frame(
                    oa.works(
                        filter=f"primary_topic.id:{tid},type:{kind},has_abstract:true",
                        sort="cited_by_count:desc",
                        max_results=per_topic,
                    )
                )
                by_kind[kind] = w[~w.id.isin(skip_works) & (w.abstract != "")]
            n = min(len(by_kind["review"]), len(by_kind["article"]))
            if n == 0:
                continue
            for kind, label in (("review", 1), ("article", 0)):
                frames.append(by_kind[kind].head(n).assign(label=label, topic_id=tid))
            used += 1
    if not frames:
        return pd.DataFrame(columns=[*COLUMNS, "label", "topic_id"])
    return pd.concat(frames, ignore_index=True)


def zscore(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def start_here_score(relevance, p_intro, citations) -> np.ndarray:
    """«Empieza aquí» = z(relevancia) + z(P(introductorio)) + z(log(1 + citas)), estandarizadas
    dentro de los candidatos. Pesos iguales, fijados antes de ver test (spec §4.2)."""
    return (
        zscore(relevance) + zscore(p_intro) + zscore(np.log1p(np.asarray(citations, dtype=float)))
    )


def rank_ids(ids: list[str], scores) -> list[str]:
    """Orden descendente estable (en empate se respeta el orden de OpenAlex)."""
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    return [ids[i] for i in order]


def ranking_orders(
    cands: pd.DataFrame, query: str, encoder: Encoder, p_intro
) -> dict[str, list[str]]:
    ids = cands.id.tolist()
    if not ids:
        return {name: [] for name in ORDERS}
    texts = cands.text.tolist()
    bm25 = BM25().fit([simple_tokenize(t) for t in texts]).scores(simple_tokenize(query))
    dense = cosine_scores(encoder([query], "query")[0], encoder(texts, "passage"))
    citations = cands.citations.to_numpy(dtype=float)
    return {
        "openalex": ids,
        "citas": rank_ids(ids, citations),
        "bm25": rank_ids(ids, bm25),
        "denso": rank_ids(ids, dense),
        "combinado": rank_ids(ids, start_here_score(dense, p_intro, citations)),
    }


def topic_metrics(cands: pd.DataFrame, gold: frozenset[str], orders: dict, k: int = 20) -> dict:
    ids = set(cands.id)
    return {
        "ceiling": len(ids & gold) / len(gold),
        "abstract_frac": float((cands.abstract != "").mean()) if len(cands) else 0.0,
        "n_candidates": len(cands),
        "orders": {
            name: {"recall": recall_at_k(r, set(gold), k), "ndcg": ndcg_at_k(r, set(gold), k)}
            for name, r in orders.items()
        },
    }
