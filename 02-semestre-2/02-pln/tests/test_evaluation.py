"""Métricas propias contra scikit-learn y statsmodels; bootstrap por conglomerados."""

import numpy as np
import pytest
from research_paths.evaluation import (
    average_precision,
    best_threshold,
    cluster_bootstrap,
    f1_binary,
    holm,
    ndcg_at_k,
    recall_at_k,
)
from sklearn.metrics import average_precision_score, f1_score, ndcg_score
from statsmodels.stats.multitest import multipletests


def test_f1_matches_sklearn_and_is_zero_without_true_positives():
    rng = np.random.default_rng(0)
    for _ in range(50):
        y, p = rng.integers(0, 2, 40), rng.integers(0, 2, 40)
        assert f1_binary(y, p) == pytest.approx(f1_score(y, p, zero_division=0))
    assert f1_binary(np.zeros(5), np.zeros(5)) == 0.0


def test_average_precision_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    for _ in range(100):
        y = rng.integers(0, 2, 30)
        y[0] = 1
        s = np.round(rng.random(30), 1)  # muchos empates
        assert average_precision(y, s) == pytest.approx(average_precision_score(y, s))


def test_average_precision_without_positives_is_zero_not_nan():
    assert average_precision(np.zeros(4), np.arange(4.0)) == 0.0


def test_best_threshold_maximizes_f1():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.8, 0.7, 0.75, 0.9])
    t = best_threshold(y, s)
    best = max(f1_binary(y, s >= c) for c in np.unique(s))
    assert f1_binary(y, s >= t) == best


def test_best_threshold_without_positives_predicts_nothing():
    assert best_threshold(np.zeros(3), np.array([0.1, 0.5, 0.9])) == np.inf


def test_recall_and_ndcg_at_k():
    ranked = ["a", "x", "b", "y"]
    assert recall_at_k(ranked, {"a", "b", "z"}, 2) == pytest.approx(1 / 3)
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0
    with pytest.raises(ValueError):
        recall_at_k(ranked, set(), 2)


def test_ndcg_matches_sklearn_when_all_relevant_are_ranked():
    rng = np.random.default_rng(2)
    for _ in range(50):
        ids = [f"w{i}" for i in range(15)]
        rel = {i for i in ids if rng.random() < 0.3} or {"w3"}
        scores = np.sort(rng.random(15))[::-1]
        y_true = np.array([[1.0 if i in rel else 0.0 for i in ids]])
        ref = ndcg_score(y_true, scores[None, :], k=10)
        if len(rel) <= 10:
            assert ndcg_at_k(ids, rel, 10) == pytest.approx(ref)


def test_holm_matches_statsmodels():
    p = [0.01, 0.04, 0.03, 0.2]
    np.testing.assert_allclose(holm(p), multipletests(p, method="holm")[1])


def test_cluster_bootstrap_resamples_whole_clusters():
    clusters = np.array([0, 1, 1, 1])
    out = cluster_bootstrap(clusters, lambda idx: float(len(idx)), np.random.default_rng(0), 200)
    assert out["delta"] == 4.0
    assert set(np.unique(out["_boots"])) <= {2.0, 4.0, 6.0}


def test_cluster_bootstrap_p_value_floor_and_direction():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(30), 3)
    diff = rng.normal(1.0, 0.1, 90)
    out = cluster_bootstrap(clusters, lambda idx: float(diff[idx].mean()), rng, 2000)
    assert out["p"] == pytest.approx(1 / 2000)
    assert out["ci"][0] < out["delta"] < out["ci"][1]
