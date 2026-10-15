import numpy as np
import pytest
from applied_stats import benjamini_hochberg, bonferroni, holm
from scipy import stats

P = np.array([0.01, 0.04, 0.03, 0.005, 0.20])


def test_bonferroni():
    np.testing.assert_allclose(bonferroni(P), [0.05, 0.2, 0.15, 0.025, 1.0])


def test_holm_hand_computed():
    # orden: 0.005·5, 0.01·4, 0.03·3, 0.04·2, 0.20·1 → acumulado máximo
    np.testing.assert_allclose(holm(P), [0.04, 0.09, 0.09, 0.025, 0.2])


def test_bh_matches_scipy():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=50) ** 2
    np.testing.assert_allclose(benjamini_hochberg(p), stats.false_discovery_control(p))


def test_holm_never_less_powerful_than_bonferroni():
    p = np.random.default_rng(1).uniform(size=30) ** 3
    assert np.all(holm(p) <= bonferroni(p) + 1e-15)


def test_fwer_control_by_simulation():
    rng = np.random.default_rng(2)
    p = rng.uniform(size=(4000, 10))  # todas las H₀ ciertas
    any_rejection = np.array([np.any(holm(row) <= 0.05) for row in p])
    assert any_rejection.mean() == pytest.approx(0.05, abs=0.012)
