import numpy as np
import pytest
from applied_ml import DecisionTreeClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="module")
def split():
    X, y = make_moons(800, noise=0.35, random_state=0)
    return train_test_split(X, y, test_size=0.5, random_state=0)


def test_forest_beats_single_deep_tree(split):
    Xtr, Xte, ytr, yte = split
    tree = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr).score(Xte, yte)
    forest = RandomForestClassifier(n_estimators=60, max_features=None, random_state=0).fit(
        Xtr, ytr
    )
    assert forest.score(Xte, yte) > tree


def test_oob_score_estimates_test_accuracy(split):
    Xtr, Xte, ytr, yte = split
    forest = RandomForestClassifier(n_estimators=80, oob_score=True, random_state=1).fit(Xtr, ytr)
    assert forest.oob_score_ == pytest.approx(forest.score(Xte, yte), abs=0.05)


def test_forest_is_reproducible(split):
    Xtr, Xte, ytr, _ = split
    a = RandomForestClassifier(n_estimators=10, random_state=3).fit(Xtr, ytr).predict_proba(Xte)
    b = RandomForestClassifier(n_estimators=10, random_state=3).fit(Xtr, ytr).predict_proba(Xte)
    np.testing.assert_array_equal(a, b)


def test_boosting_train_loss_decreases_monotonically(split):
    Xtr, _, ytr, _ = split
    gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=2).fit(Xtr, ytr)
    assert np.all(np.diff(gb.train_loss_) <= 1e-9)


def test_boosting_competitive_with_sklearn():
    from sklearn.ensemble import GradientBoostingClassifier as SkGB

    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    kw = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 2}
    ours = roc_auc_score(
        yte, GradientBoostingClassifier(**kw).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    )
    ref = roc_auc_score(yte, SkGB(**kw, random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    assert ours == pytest.approx(ref, abs=0.01)


def test_staged_predictions_end_at_final(split):
    Xtr, Xte, ytr, _ = split
    gb = GradientBoostingClassifier(n_estimators=20).fit(Xtr, ytr)
    *_, last = gb.staged_decision_function(Xte)
    np.testing.assert_allclose(last, gb.decision_function(Xte))
