import numpy as np
import pytest
from applied_ml.metrics import roc_auc
from applied_ml.model_selection import bootstrap_metric_ci, cross_validate, stratified_kfold
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def test_stratified_folds_partition_and_preserve_ratio():
    y = np.array([0] * 90 + [1] * 10)
    folds = stratified_kfold(y, 5)
    tests = np.concatenate([te for _, te in folds])
    assert sorted(tests) == list(range(100))
    for train, test in folds:
        assert set(train).isdisjoint(test)
        assert y[test].sum() == 2


def test_cross_validate_matches_sklearn_with_same_folds():
    X, y = load_breast_cancer(return_X_y=True)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
    ours = cross_validate(pipe, X, y, lambda m, X, y: m.score(X, y), folds)
    ref = cross_val_score(pipe, X, y, cv=folds)
    np.testing.assert_allclose(ours, ref)


def test_bootstrap_ci_covers_and_shrinks_with_n():
    rng = np.random.default_rng(0)

    def sample(n):
        y = rng.integers(0, 2, n)
        return y, rng.normal(y * 1.0, 1.0)

    y1, s1 = sample(300)
    y2, s2 = sample(5000)
    v1, lo1, hi1 = bootstrap_metric_ci(roc_auc, y1, s1, n_boot=500)
    _, lo2, hi2 = bootstrap_metric_ci(roc_auc, y2, s2, n_boot=500)
    assert lo1 < v1 < hi1
    assert (hi2 - lo2) < (hi1 - lo1)
    assert lo2 < 0.76 < hi2  # AUC teórica = Φ(1/√2) ≈ 0.760


def test_bootstrap_ci_reproducible():
    y, s = np.array([0, 1] * 50), np.linspace(0, 1, 100)
    a = bootstrap_metric_ci(roc_auc, y, s, n_boot=200, rng=np.random.default_rng(1))
    b = bootstrap_metric_ci(roc_auc, y, s, n_boot=200, rng=np.random.default_rng(1))
    assert a == pytest.approx(b)
