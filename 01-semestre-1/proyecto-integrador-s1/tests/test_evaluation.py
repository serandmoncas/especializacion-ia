import numpy as np
import pytest
from book_topics.evaluation import (
    cluster_bootstrap_diff,
    multilabel_report,
    per_user_diff_bootstrap,
    precision_at_k,
)
from sklearn.metrics import f1_score


def test_multilabel_report_matches_sklearn():
    rng = np.random.default_rng(0)
    yt, yp = rng.random((50, 6)) < 0.3, rng.random((50, 6)) < 0.3
    rep = multilabel_report(yt, yp)
    assert rep["micro_f1"] == pytest.approx(f1_score(yt, yp, average="micro", zero_division=0))
    assert rep["macro_f1"] == pytest.approx(f1_score(yt, yp, average="macro", zero_division=0))


def test_precision_at_k():
    yt = np.array([[1, 0, 1, 0]], bool)
    assert precision_at_k(yt, np.array([[0.9, 0.8, 0.7, 0.1]]), k=3) == pytest.approx(2 / 3)


def test_cluster_bootstrap_wider_than_naive_with_correlated_groups():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(40), 25)
    effect = rng.normal(0, 1, 40)[groups]
    a, b = effect + rng.normal(0, 0.2, len(groups)), np.zeros(len(groups))
    clustered = cluster_bootstrap_diff(lambda i: a[i].mean(), lambda i: b[i].mean(), groups)
    naive = cluster_bootstrap_diff(lambda i: a[i].mean(), lambda i: b[i].mean(), np.arange(len(a)))

    def width(r):
        return r["ci"][1] - r["ci"][0]

    assert width(clustered) > 2 * width(naive)
    assert clustered["ci"][0] < a.mean() < clustered["ci"][1]


def test_bootstrap_p_value_small_for_clear_difference():
    rng = np.random.default_rng(2)
    res = per_user_diff_bootstrap(np.full(200, 0.5), np.full(200, 0.3) + rng.normal(0, 0.01, 200))
    assert res["diff"] == pytest.approx(0.2, abs=0.01) and res["p_value"] < 0.001
