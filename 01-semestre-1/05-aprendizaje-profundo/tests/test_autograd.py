import numpy as np
import pytest
from applied_dl import Tensor, cross_entropy
from applied_math import numerical_gradient

RNG = np.random.default_rng(0)


def check_grad(fn, *shapes, positive=False, tol=1e-6):
    """Compara el gradiente del autograd con diferencias centrales para cada entrada."""
    arrays = [RNG.normal(size=s) for s in shapes]
    if positive:
        arrays = [np.abs(a) + 0.5 for a in arrays]
    tensors = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    fn(*tensors).backward()
    for i, t in enumerate(tensors):

        def scalar(x, i=i):
            args = [Tensor(a) for a in arrays]
            args[i] = Tensor(x)
            return float(fn(*args).data)

        num = numerical_gradient(scalar, arrays[i].copy())
        np.testing.assert_allclose(t.grad, num, atol=tol, rtol=1e-5)


@pytest.mark.parametrize(
    ("name", "fn", "shapes", "positive"),
    [
        ("add+broadcast", lambda a, b: (a + b).sum(), [(3, 4), (4,)], False),
        ("mul+broadcast", lambda a, b: (a * b).sum(), [(3, 4), (3, 1)], False),
        ("sub/neg", lambda a, b: (a - b).sum(), [(2, 3), (2, 3)], False),
        ("div", lambda a, b: (a / b).sum(), [(2, 3), (2, 3)], True),
        ("pow", lambda a: (a**3).sum(), [(5,)], False),
        ("matmul", lambda a, b: (a @ b).sum(), [(3, 4), (4, 2)], False),
        ("relu", lambda a: (a.relu() * a).sum(), [(10,)], False),
        ("tanh", lambda a: a.tanh().sum(), [(6,)], False),
        ("sigmoid", lambda a: a.sigmoid().sum(), [(6,)], False),
        ("exp/log", lambda a: (a.exp() + a.log()).sum(), [(4,)], True),
        ("sum axis", lambda a: (a.sum(axis=0) ** 2).sum(), [(3, 4)], False),
        (
            "mean keepdims",
            lambda a: ((a - a.mean(axis=1, keepdims=True)) ** 2).sum(),
            [(3, 4)],
            False,
        ),
        ("reshape", lambda a: (a.reshape(6, 2) @ Tensor(np.ones((2, 1)))).sum(), [(3, 4)], False),
        (
            "getitem",
            lambda a: (a[np.array([0, 0, 2]), np.array([1, 1, 3])] ** 2).sum(),
            [(3, 4)],
            False,
        ),
        (
            "log_softmax",
            lambda a: (a.log_softmax() * Tensor(np.arange(4.0))).sum(),
            [(3, 4)],
            False,
        ),
    ],
)
def test_op_gradients(name, fn, shapes, positive):
    check_grad(fn, *shapes, positive=positive)


def test_cross_entropy_gradient_and_value():
    logits = RNG.normal(size=(5, 3))
    y = np.array([0, 2, 1, 1, 0])
    t = Tensor(logits.copy(), requires_grad=True)
    loss = cross_entropy(t, y)
    loss.backward()
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    assert float(loss.data) == pytest.approx(-np.mean(np.log(p[np.arange(5), y])))
    np.testing.assert_allclose(t.grad, (p - np.eye(3)[y]) / 5, atol=1e-12)


def test_cross_entropy_is_stable_with_huge_logits():
    t = Tensor(np.array([[1000.0, 0.0, -1000.0]]), requires_grad=True)
    loss = cross_entropy(t, np.array([0]))
    loss.backward()
    assert np.isfinite(loss.data) and np.all(np.isfinite(t.grad))


def test_shared_subexpression_accumulates_gradient():
    x = Tensor(np.array(3.0), requires_grad=True)
    y = x * x + x  # x se usa tres veces
    y.backward()
    assert x.grad == pytest.approx(2 * 3 + 1)


def test_deep_graph_does_not_hit_recursion_limit():
    x = Tensor(np.array(1.0), requires_grad=True)
    y = x
    for _ in range(5000):
        y = y * 1.0001
    y.backward()
    assert x.grad == pytest.approx(1.0001**5000)


def test_no_grad_for_constants():
    a = Tensor(np.ones(3), requires_grad=True)
    c = Tensor(np.ones(3))
    (a * c).sum().backward()
    assert c.grad is None and a.grad is not None


def test_intermediate_constant_nodes_are_skipped():
    # Regresión: x − c crea el nodo intermedio −c (sin gradiente); backward fallaba con None.
    x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    c = Tensor(np.array([0.5, 0.5]))
    ((x - c) ** 2).sum().backward()
    np.testing.assert_allclose(x.grad, 2 * (x.data - c.data))
