import numpy as np
import pytest
from applied_stats import (
    cuped,
    power_two_proportions,
    sample_ratio_mismatch,
    sample_size_two_proportions,
    two_proportion_ztest,
)


def test_sample_size_reference_value():
    # Valor de referencia de calculadoras estándar para 10 % → 12 %, α = .05, potencia .8
    assert sample_size_two_proportions(0.10, 0.12) == pytest.approx(3841, abs=2)


def test_power_and_sample_size_are_consistent():
    n = sample_size_two_proportions(0.05, 0.06, power=0.9)
    assert power_two_proportions(0.05, 0.06, n) == pytest.approx(0.9, abs=0.005)


def test_power_by_simulation():
    rng = np.random.default_rng(0)
    p1, p2, n, reps = 0.10, 0.12, 3841, 3000
    sa, sb = rng.binomial(n, p1, size=reps), rng.binomial(n, p2, size=reps)
    rejected = np.mean(
        [two_proportion_ztest(x, n, y, n).pvalue < 0.05 for x, y in zip(sa, sb, strict=True)]
    )
    assert rejected == pytest.approx(0.8, abs=0.025)


def test_srm_detects_imbalance():
    _, p_ok = sample_ratio_mismatch([10_050, 9_950])
    _, p_bad = sample_ratio_mismatch([10_500, 9_500])
    _, p_ratio = sample_ratio_mismatch([20_000, 10_000], expected_ratios=[2, 1])
    assert p_ok > 0.1
    assert p_bad < 1e-10
    assert p_ratio == pytest.approx(1.0)


def test_cuped_preserves_mean_and_reduces_variance():
    rng = np.random.default_rng(1)
    x = rng.normal(size=10_000)
    y = 2 * x + rng.normal(size=10_000)  # ρ² = 4/5
    y_adj = cuped(y, x)
    assert y_adj.mean() == pytest.approx(y.mean())
    assert y_adj.var() / y.var() == pytest.approx(1 - 0.8, abs=0.02)


def test_cuped_does_not_change_treatment_effect():
    rng = np.random.default_rng(2)
    n = 20_000
    x = rng.normal(size=n)
    treat = rng.integers(0, 2, size=n).astype(bool)
    y = x + 0.1 * treat + rng.normal(size=n)
    y_adj = cuped(y, x)
    effect_raw = y[treat].mean() - y[~treat].mean()
    effect_adj = y_adj[treat].mean() - y_adj[~treat].mean()
    assert effect_adj == pytest.approx(0.1, abs=0.03)
    assert y_adj[treat].var() < y[treat].var()
    assert abs(effect_adj - 0.1) < abs(effect_raw - 0.1) + 0.03
