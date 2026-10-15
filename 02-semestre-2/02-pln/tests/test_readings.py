"""Lecturas: temas con revisión, candidatos sin futuro, conjunto de entrenamiento y órdenes."""

import numpy as np
import pytest
from research_paths.readings import (
    ORDERS,
    EvalTopic,
    intro_training_set,
    ranking_orders,
    sample_eval_topics,
    start_here_score,
    topic_candidates,
    topic_metrics,
    works_frame,
    zscore,
)
from research_paths.retrieval import hashing_encoder


def test_works_frame_tolerates_null_fields():
    df = works_frame(
        [{"id": None, "title": None, "abstract_inverted_index": None, "publication_year": None,
          "cited_by_count": None, "type": None}]
    )  # fmt: skip
    row = df.iloc[0]
    assert (row.title, row.abstract, row.text, row.year, row.citations) == ("", "", "", 0, 0)


def test_sample_eval_topics_skips_topics_without_review_and_is_reproducible(oa_world):
    topics = sample_eval_topics(oa_world, fields=("17", "26"), per_field=5, seed=42)
    assert sorted(t.topic_id for t in topics) == ["T1", "T2", "T4", "T5"]
    t1 = next(t for t in topics if t.topic_id == "T1")
    assert t1.review_id == "R1" and t1.review_year == 2020 and len(t1.gold) == 25
    assert topics == sample_eval_topics(oa_world, fields=("17", "26"), per_field=5, seed=42)
    assert len(sample_eval_topics(oa_world, fields=("17",), per_field=1, seed=42)) == 1


def test_candidates_exclude_the_review_and_the_future(oa_world):
    topic = EvalTopic("T1", "Tema 1", "17", "R1", 2010, frozenset({"W100"}))
    cands = topic_candidates(oa_world, topic)
    assert len(cands) > 0 and cands.year.max() < 2010 and "R1" not in set(cands.id)
    assert list(cands.relevance_rank) == list(range(len(cands)))


def test_intro_training_set_is_balanced_and_excludes_evaluation(oa_world):
    evals = sample_eval_topics(oa_world, fields=("17", "26"), per_field=1, seed=42)
    df = intro_training_set(oa_world, evals, fields=("17", "26"), topics_per_field=3, per_topic=5)
    assert not set(df.topic_id) & {t.topic_id for t in evals}
    gold = set().union(*(t.gold for t in evals))
    assert not set(df.id) & gold
    assert (df.abstract != "").all()
    counts = df.groupby(["topic_id", "label"]).size().unstack()
    assert (counts[0] == counts[1]).all()


def test_start_here_score_is_sum_of_zscores():
    rel, p, cites = np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5, 0.5]), np.array([0, 9, 99])
    expected = zscore(rel) + zscore(np.log1p(cites))
    np.testing.assert_allclose(start_here_score(rel, p, cites), expected)
    assert np.all(zscore(np.ones(3)) == 0)


def test_ranking_orders(oa_world):
    topic = EvalTopic("T1", "Tema 1", "17", "R1", 2020, frozenset())
    cands = topic_candidates(oa_world, topic)
    p = np.linspace(0, 1, len(cands))
    orders = ranking_orders(cands, "tema 1 revisión", hashing_encoder(), p)
    assert set(orders) == set(ORDERS)
    assert orders["openalex"] == list(cands.id)
    cites = dict(zip(cands.id, cands.citations, strict=True))
    assert [cites[i] for i in orders["citas"]] == sorted(cands.citations, reverse=True)
    for ranked in orders.values():
        assert sorted(ranked) == sorted(cands.id)


def test_ranking_orders_and_metrics_with_no_candidates():
    empty = works_frame([])
    orders = ranking_orders(empty, "x", hashing_encoder(), np.zeros(0))
    m = topic_metrics(empty, frozenset({"W1"}), orders)
    assert m["ceiling"] == 0 and m["abstract_frac"] == 0
    assert all(v == {"recall": 0.0, "ndcg": 0.0} for v in m["orders"].values())


def test_topic_metrics_ceiling(oa_world):
    topic = EvalTopic("T1", "Tema 1", "17", "R1", 2020, frozenset({"W100", "W101", "NO_ESTA"}))
    cands = topic_candidates(oa_world, topic)
    orders = ranking_orders(cands, "tema", hashing_encoder(), np.zeros(len(cands)))
    m = topic_metrics(cands, topic.gold, orders)
    assert m["ceiling"] == pytest.approx(2 / 3)
    assert m["orders"]["citas"]["recall"] == pytest.approx(2 / 3)  # W100 y W101 son los más citados
