import numpy as np
import pytest
from applied_ml import DecisionTreeClassifier, DecisionTreeRegressor
from applied_ml.tree import _entropy, _gini
from sklearn import tree as sk_tree
from sklearn.datasets import load_breast_cancer, make_moons, make_regression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def test_impurity_functions():
    assert _gini(np.array([5, 5])) == pytest.approx(0.5)
    assert _gini(np.array([10, 0])) == pytest.approx(0.0)
    assert _entropy(np.array([5, 5])) == pytest.approx(1.0)
    assert _entropy(np.array([8, 0])) == pytest.approx(0.0)


def test_xor_needs_depth_two():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 5, dtype=float)
    y = np.array([0, 1, 1, 0] * 5)
    assert DecisionTreeClassifier(max_depth=2).fit(X, y).score(X, y) == 1.0


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_root_split_matches_sklearn(criterion):
    X, y = load_breast_cancer(return_X_y=True)
    ours = DecisionTreeClassifier(criterion=criterion, max_depth=1).fit(X, y)
    ref = sk_tree.DecisionTreeClassifier(criterion=criterion, max_depth=1, random_state=0).fit(X, y)
    assert ours.tree_feature_[0] == ref.tree_.feature[0]
    assert ours.tree_threshold_[0] == pytest.approx(ref.tree_.threshold[0])


def test_full_depth_tree_memorizes_training_set():
    X, y = make_moons(300, noise=0.3, random_state=0)
    t = DecisionTreeClassifier().fit(X, y)
    assert t.score(X, y) == 1.0
    assert t.get_depth() > 5


def test_generalization_close_to_sklearn():
    X, y = load_breast_cancer(return_X_y=True)
    kw = {"max_depth": 4, "min_samples_leaf": 5}
    ours = cross_val_score(DecisionTreeClassifier(**kw), X, y, cv=5).mean()
    ref = cross_val_score(sk_tree.DecisionTreeClassifier(**kw, random_state=0), X, y, cv=5).mean()
    assert ours == pytest.approx(ref, abs=0.02)


def test_min_samples_leaf_respected():
    X, y = make_moons(200, noise=0.3, random_state=1)
    t = DecisionTreeClassifier(min_samples_leaf=15).fit(X, y)
    leaves = t.tree_feature_ == -1
    assert t.tree_n_[leaves].min() >= 15


def test_apply_and_predict_proba_consistent():
    X, y = make_moons(200, noise=0.2, random_state=2)
    t = DecisionTreeClassifier(max_depth=3).fit(X, y)
    leaves = t.apply(X)
    np.testing.assert_allclose(t.predict_proba(X), t.tree_value_[leaves])
    np.testing.assert_allclose(t.predict_proba(X).sum(axis=1), 1.0)


def test_feature_importances_sum_to_one_and_find_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 5))
    y = (X[:, 2] > 0).astype(int)
    imp = DecisionTreeClassifier(max_depth=3).fit(X, y).feature_importances_
    assert imp.sum() == pytest.approx(1.0)
    assert imp.argmax() == 2


def test_regressor_matches_sklearn_predictions():
    X, y = make_regression(300, 4, noise=5.0, random_state=0)
    ours = DecisionTreeRegressor(max_depth=4).fit(X, y)
    ref = sk_tree.DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)
    np.testing.assert_allclose(ours.predict(X), ref.predict(X), rtol=1e-8)


def test_works_inside_sklearn_pipeline():
    X, y = load_breast_cancer(return_X_y=True)
    pipe = make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=3))
    assert cross_val_score(pipe, X, y, cv=3).mean() > 0.88
