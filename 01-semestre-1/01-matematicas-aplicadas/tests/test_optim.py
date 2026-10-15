import numpy as np
from applied_math import gradient_descent, minibatch_sgd, mse, mse_gradient

A = np.diag([1.0, 10.0])  # número de condición κ = 10


def f(x):
    return 0.5 * x @ A @ x


def grad_f(x):
    return A @ x


def test_gd_converges_on_convex_quadratic():
    res = gradient_descent(f, grad_f, np.array([5.0, 5.0]), lr=0.1)
    assert res.converged
    np.testing.assert_allclose(res.x, 0, atol=1e-6)


def test_gd_diverges_when_lr_exceeds_two_over_lmax():
    # GD es estable en cuadráticas sii lr < 2/λ_max = 0.2
    res = gradient_descent(f, grad_f, np.array([1.0, 1.0]), lr=0.25, max_iter=2000)
    assert not res.converged


def test_momentum_reduces_iterations():
    x0 = np.array([5.0, 5.0])
    plain = gradient_descent(f, grad_f, x0, lr=0.05)
    heavy = gradient_descent(f, grad_f, x0, lr=0.05, momentum=0.7)
    assert heavy.converged and plain.converged
    assert heavy.n_iter < plain.n_iter


def test_minibatch_sgd_reduces_loss_and_is_reproducible():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(scale=0.1, size=300)
    w0 = np.zeros(3)
    a = minibatch_sgd(mse, mse_gradient, w0, X, y, lr=0.05, rng=np.random.default_rng(1))
    b = minibatch_sgd(mse, mse_gradient, w0, X, y, lr=0.05, rng=np.random.default_rng(1))
    assert a.history[-1] < 0.05 * a.history[0]
    np.testing.assert_array_equal(a.x, b.x)
