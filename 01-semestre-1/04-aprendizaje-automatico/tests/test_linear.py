import numpy as np
import pytest
from applied_ml import LogisticRegression
from sklearn import linear_model
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


@pytest.fixture(scope="module")
def data():
    X, y = load_breast_cancer(return_X_y=True)
    return StandardScaler().fit_transform(X), y


@pytest.mark.parametrize("solver", ["gd", "newton"])
@pytest.mark.parametrize("alpha", [1e-2, 1e-1])
def test_matches_sklearn_objective(data, alpha, solver):
    X, y = data
    ours = LogisticRegression(alpha=alpha, solver=solver).fit(X, y)
    ref = linear_model.LogisticRegression(C=1 / (len(y) * alpha), tol=1e-10, max_iter=10_000).fit(
        X, y
    )
    assert ours.converged_
    np.testing.assert_allclose(ours.coef_, ref.coef_, atol=2e-3)
    assert ours.intercept_[0] == pytest.approx(ref.intercept_[0], abs=2e-3)
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), atol=1e-3)


def test_loss_decreases_and_labels_are_preserved(data):
    X, y = data
    labels = np.where(y == 1, "benigno", "maligno")
    model = LogisticRegression(alpha=1e-2).fit(X, labels)
    assert set(model.predict(X)) <= {"benigno", "maligno"}
    assert model.score(X, labels) > 0.95
    h = model.loss_history_
    assert h[-1] < h[0]


def test_rejects_multiclass():
    with pytest.raises(ValueError):
        LogisticRegression().fit(np.zeros((3, 1)), [0, 1, 2])


def test_newton_converges_in_few_iterations_where_gd_struggles():
    # Features con escalas muy distintas: κ grande. Newton no se entera; GD sí.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5)) * np.array([1, 1, 1, 0.01, 0.01])
    y = (X @ np.array([1, -1, 0.5, 50, -50]) + rng.normal(size=2000) > 0).astype(int)
    newton = LogisticRegression(alpha=1e-4, solver="newton").fit(X, y)
    gd = LogisticRegression(alpha=1e-4, solver="gd", max_iter=2000).fit(X, y)
    assert newton.converged_ and newton.n_iter_ <= 15
    assert not gd.converged_
    assert newton.loss_history_[-1] < gd.loss_history_[-1]


def test_unknown_solver():
    with pytest.raises(ValueError):
        LogisticRegression(solver="sgd").fit(np.zeros((4, 1)), [0, 1, 0, 1])
