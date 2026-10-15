import numpy as np
import pytest
from applied_math import pca, power_iteration, project_onto


def test_projection_residual_is_orthogonal():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 2))
    v = rng.normal(size=6)
    p = project_onto(v, A)
    np.testing.assert_allclose(A.T @ (v - p), 0, atol=1e-10)


def test_projection_is_idempotent():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 3))
    v = rng.normal(size=5)
    p = project_onto(v, A)
    np.testing.assert_allclose(project_onto(p, A), p, atol=1e-10)


def test_power_iteration_matches_eigh():
    rng = np.random.default_rng(2)
    B = rng.normal(size=(5, 5))
    M = B @ B.T  # simétrica PSD: autovalores reales y no negativos
    eigval, eigvec = power_iteration(M)
    expected = np.linalg.eigvalsh(M)[-1]
    assert eigval == pytest.approx(expected, rel=1e-6)
    np.testing.assert_allclose(M @ eigvec, eigval * eigvec, atol=1e-4)


def test_pca_recovers_dominant_direction():
    rng = np.random.default_rng(3)
    t = rng.normal(size=500)
    direction = np.array([3.0, 4.0]) / 5.0
    X = np.outer(t, direction) * 5 + rng.normal(scale=0.1, size=(500, 2))
    _, components, var = pca(X, 1)
    assert abs(components[0] @ direction) == pytest.approx(1.0, abs=1e-3)
    assert var[0] > 20


def test_pca_variance_matches_covariance_eigenvalues():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 4)) @ rng.normal(size=(4, 4))
    _, _, var = pca(X, 4)
    expected = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False)))[::-1]
    np.testing.assert_allclose(var, expected, rtol=1e-8)
