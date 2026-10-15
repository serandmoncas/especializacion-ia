import numpy as np
import pytest
from applied_recsys.metrics import (
    catalog_coverage,
    mean_popularity_percentile,
    ranking_metrics,
    top_n,
)
from scipy import sparse
from sklearn.metrics import ndcg_score


def test_top_n_excludes_seen_and_orders():
    scores = np.array([[0.1, 0.9, 0.5, 0.7], [0.4, 0.3, 0.2, 0.1]])
    seen = sparse.csr_matrix(np.array([[0, 1, 0, 0], [0, 0, 0, 0]]))
    recs = top_n(scores, seen, 2)
    assert recs.tolist() == [[3, 2], [0, 1]]


def test_ranking_metrics_hand_example():
    recs = np.array([[5, 1, 7, 3]])
    m = ranking_metrics(recs, [{1, 3, 9}], k=4)
    assert m["precision"][0] == pytest.approx(0.5)
    assert m["recall"][0] == pytest.approx(2 / 3)
    dcg = 1 / np.log2(3) + 1 / np.log2(5)
    idcg = 1 + 1 / np.log2(3) + 1 / np.log2(4)
    assert m["ndcg"][0] == pytest.approx(dcg / idcg)
    assert m["hit"][0] == 1.0


def test_ndcg_matches_sklearn_binary():
    rng = np.random.default_rng(0)
    for _ in range(20):
        scores = rng.random(30)
        rel = set(rng.choice(30, 4, replace=False).tolist())
        recs = np.argsort(-scores)[None, :10]
        ours = ranking_metrics(recs, [rel], k=10)["ndcg"][0]
        truth = np.zeros(30)
        truth[list(rel)] = 1
        assert ours == pytest.approx(ndcg_score(truth[None, :], scores[None, :], k=10))


def test_coverage_and_popularity_bias():
    pop = np.array([100.0, 50, 10, 1])
    assert catalog_coverage(np.array([[0, 1], [0, 1]]), 4) == 0.5
    assert mean_popularity_percentile(np.array([[0]]), pop) == pytest.approx(100.0)
    assert mean_popularity_percentile(np.array([[3]]), pop) == pytest.approx(0.0)
