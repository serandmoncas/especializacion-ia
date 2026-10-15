import numpy as np
import pytest
from applied_math import LinearRegressionGD, fit_normal_equation, gradient_check, mse, mse_gradient
from sklearn.linear_model import LinearRegression


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(400, 4))
    y = 3.0 + X @ np.array([1.5, -2.0, 0.0, 0.7]) + rng.normal(scale=0.3, size=400)
    return X, y


def test_mse_gradient_is_correct(data):
    X, y = data
    w = np.random.default_rng(0).normal(size=X.shape[1])
    assert gradient_check(lambda v: mse(v, X, y), lambda v: mse_gradient(v, X, y), w) < 1e-7


def test_normal_equation_matches_sklearn(data):
    X, y = data
    w = fit_normal_equation(X, y)
    ref = LinearRegression().fit(X, y)
    assert w[0] == pytest.approx(ref.intercept_)
    np.testing.assert_allclose(w[1:], ref.coef_, rtol=1e-10)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"solver": "gd", "lr": 0.1},
        {"solver": "gd", "lr": 0.05, "momentum": 0.8},
        {"solver": "sgd", "lr": 0.02, "epochs": 200, "batch_size": 16},
    ],
)
def test_gd_solvers_match_closed_form(data, kwargs):
    X, y = data
    w = fit_normal_equation(X, y)
    model = LinearRegressionGD(**kwargs).fit(X, y)
    # SGD con lr constante oscila alrededor del óptimo: tolerancia más laxa
    atol = 5e-2 if kwargs["solver"] == "sgd" else 1e-5
    assert model.intercept_ == pytest.approx(w[0], abs=atol)
    np.testing.assert_allclose(model.coef_, w[1:], atol=atol)


def test_unknown_solver_raises(data):
    X, y = data
    with pytest.raises(ValueError):
        LinearRegressionGD(solver="adam").fit(X, y)  # type: ignore[arg-type]
