import numpy as np
import pytest
from applied_cv.cocoa.evaluation import (
    bootstrap_diff,
    bootstrap_unpaired,
    cluster_members,
    decide,
    holm,
    macro_f1,
    match_boxes,
    recall_above,
    resample,
    sick_score,
    threshold_at_fpr,
)


def test_threshold_at_fpr_controls_false_alarms():
    negatives = np.arange(100) / 100
    thr = threshold_at_fpr(negatives, fpr=0.1)
    assert recall_above(negatives, thr) <= 0.1
    assert recall_above(negatives, thr) == pytest.approx(0.09)
    assert threshold_at_fpr(np.array([]), 0.1) == 0.5
    assert np.isnan(recall_above(np.array([]), 0.5))


def test_sick_score_and_macro_f1():
    proba = np.array([[0.9, 0.1, 0, 0], [0.2, 0.7, 0.1, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    np.testing.assert_allclose(sick_score(proba), [0.1, 0.8, 1.0, 1.0])
    assert macro_f1(np.array([0, 1, 2, 3]), proba) == 1.0


def test_resample_keeps_clusters_whole():
    clusters = np.array([0, 0, 0, 1, 1, 2])
    members = cluster_members(clusters)
    idx = resample(members, np.random.default_rng(0))
    counts = np.bincount(clusters[idx], minlength=3)
    assert all(c % s == 0 for c, s in zip(counts, (3, 2, 1), strict=True))


def test_bootstrap_identical_and_clear_difference():
    clusters = np.repeat(np.arange(40), 3)
    same = bootstrap_diff(lambda idx: 0.0 * len(idx), clusters, n_boot=200)
    assert same["diff"] == 0.0 and same["p_value"] == 1.0
    rng = np.random.default_rng(1)
    a, b = rng.normal(1.0, 0.2, 120), rng.normal(0.0, 0.2, 120)
    res = bootstrap_diff(lambda idx: a[idx].mean() - b[idx].mean(), clusters, n_boot=200)
    assert res["ci"][0] > 0 and res["p_value"] == pytest.approx(1 / 200)
    un = bootstrap_unpaired(
        lambda i: a[i].mean(), clusters, lambda i: b[i].mean(), clusters, n_boot=200
    )
    assert un["diff"] == pytest.approx(a.mean() - b.mean()) and un["ci"][0] > 0


def test_holm_and_decide():
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    worse = decide({"diff": -0.1, "ci": [-0.2, -0.05], "p_value": 0.001}, p_holm=0.003)
    assert worse["rechaza_h0"] and not worse["apoya_hipotesis"]


def test_match_boxes_greedy_one_to_one():
    gt = np.array([[0, 0, 10, 10], [20, 0, 30, 10], [50, 50, 60, 60]])
    pred = np.array([[1, 0, 10, 10], [0, 0, 10, 11], [21, 0, 30, 10]])
    np.testing.assert_array_equal(match_boxes(pred, gt), [1, 2, -1])
    np.testing.assert_array_equal(match_boxes(np.zeros((0, 4)), gt), [-1, -1, -1])
