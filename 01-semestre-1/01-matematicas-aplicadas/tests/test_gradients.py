import numpy as np
from applied_math import gradient_check, numerical_gradient


def test_numerical_gradient_quadratic():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, -1.0])

    def f(x):
        return 0.5 * x @ A @ x - b @ x

    x = np.array([0.3, -0.7])
    np.testing.assert_allclose(numerical_gradient(f, x), A @ x - b, atol=1e-6)


def test_numerical_gradient_does_not_mutate_input():
    x = np.array([1.0, 2.0])
    numerical_gradient(lambda z: float(z @ z), x)
    np.testing.assert_array_equal(x, [1.0, 2.0])


def test_gradient_check_detects_correct_and_wrong_gradients():
    def f(x):
        return float(np.sum(np.sin(x) * x**2))

    def good(x):
        return np.cos(x) * x**2 + 2 * x * np.sin(x)

    def bad(x):
        return np.cos(x) * x**2  # falta la regla del producto

    x = np.array([0.5, 1.5, -2.0])
    assert gradient_check(f, good, x) < 1e-7
    assert gradient_check(f, bad, x) > 1e-2
