import numpy as np
import pytest
from applied_stats import mean_ci, two_proportion_ztest, welch_ttest
from scipy import stats


@pytest.fixture
def samples():
    rng = np.random.default_rng(0)
    return rng.normal(10, 2, size=40), rng.normal(11, 5, size=60)


def test_mean_ci_matches_scipy(samples):
    a, _ = samples
    lo, hi = mean_ci(a, conf=0.9)
    ref = stats.t.interval(0.9, len(a) - 1, loc=a.mean(), scale=stats.sem(a))
    assert (lo, hi) == pytest.approx(ref)


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_welch_matches_scipy(samples, alternative):
    a, b = samples
    res = welch_ttest(a, b, alternative=alternative)
    ref = stats.ttest_ind(b, a, equal_var=False, alternative=alternative)
    assert res.statistic == pytest.approx(ref.statistic)
    assert res.pvalue == pytest.approx(ref.pvalue)


def test_welch_ci_matches_scipy(samples):
    a, b = samples
    res = welch_ttest(a, b, conf=0.95)
    ref = stats.ttest_ind(b, a, equal_var=False).confidence_interval(0.95)
    assert res.ci == pytest.approx((ref.low, ref.high))


def test_ztest_squared_equals_uncorrected_chi2():
    res = two_proportion_ztest(120, 1000, 150, 1000)
    table = [[120, 880], [150, 850]]
    chi2, p, *_ = stats.chi2_contingency(table, correction=False)
    assert res.statistic**2 == pytest.approx(chi2)
    assert res.pvalue == pytest.approx(p)
    assert res.diff == pytest.approx(0.03)


def test_ztest_type_one_error_rate_is_nominal():
    # Bajo H₀ el test debe rechazar ~5 % de las veces.
    rng = np.random.default_rng(1)
    n, p, reps = 2000, 0.1, 20_000  # SE de la tasa ≈ 0.0015
    sa, sb = rng.binomial(n, p, size=reps), rng.binomial(n, p, size=reps)
    pvals = np.array([two_proportion_ztest(x, n, y, n).pvalue for x, y in zip(sa, sb, strict=True)])
    assert np.mean(pvals < 0.05) == pytest.approx(0.05, abs=0.005)
