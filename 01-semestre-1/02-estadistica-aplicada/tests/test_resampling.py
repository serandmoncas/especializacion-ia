import numpy as np
import pytest
from applied_stats import bootstrap_ci, bootstrap_diff_ci, permutation_test, welch_ttest
from scipy import stats


def test_bootstrap_ci_close_to_scipy_percentile():
    x = np.random.default_rng(0).exponential(2.0, size=200)
    lo, hi, boots = bootstrap_ci(x, n_boot=20_000, rng=np.random.default_rng(1))
    ref = stats.bootstrap(
        (x,), np.mean, n_resamples=20_000, method="percentile", rng=np.random.default_rng(2)
    ).confidence_interval
    assert lo == pytest.approx(ref.low, rel=0.02)
    assert hi == pytest.approx(ref.high, rel=0.02)
    assert boots.std() == pytest.approx(x.std(ddof=0) / np.sqrt(len(x)), rel=0.05)


def test_bootstrap_custom_statistic_and_chunking():
    x = np.random.default_rng(3).normal(size=50_000)  # fuerza varios bloques
    lo, hi, boots = bootstrap_ci(x, lambda m: np.median(m, axis=1), n_boot=300)
    assert len(boots) == 300
    assert lo < np.median(x) < hi


def test_bootstrap_is_reproducible():
    x = np.arange(30.0)
    a = bootstrap_ci(x, n_boot=500, rng=np.random.default_rng(5))
    b = bootstrap_ci(x, n_boot=500, rng=np.random.default_rng(5))
    assert a[:2] == b[:2]


def test_bootstrap_diff_agrees_with_welch_for_large_samples():
    rng = np.random.default_rng(4)
    a, b = rng.normal(0, 1, 3000), rng.normal(0.2, 1.5, 3000)
    lo, hi, _ = bootstrap_diff_ci(a, b, n_boot=5000, rng=rng)
    w = welch_ttest(a, b)
    assert lo == pytest.approx(w.ci[0], abs=0.01)
    assert hi == pytest.approx(w.ci[1], abs=0.01)


def test_permutation_matches_scipy():
    rng = np.random.default_rng(6)
    a, b = rng.normal(0, 1, 40), rng.normal(0.5, 1, 40)
    obs, p = permutation_test(a, b, n_perm=20_000, rng=np.random.default_rng(7))
    ref = stats.permutation_test(
        (b, a),
        lambda x, y, axis: x.mean(axis=axis) - y.mean(axis=axis),
        n_resamples=20_000,
        rng=np.random.default_rng(8),
    )
    assert obs == pytest.approx(ref.statistic)
    assert p == pytest.approx(ref.pvalue, abs=0.005)


def test_permutation_pvalue_is_never_zero():
    _, p = permutation_test(np.zeros(20), np.full(20, 10.0), n_perm=999)
    assert p == pytest.approx(1 / 1000)


@pytest.mark.parametrize(("alternative", "small"), [("greater", True), ("less", False)])
def test_permutation_one_sided(alternative, small):
    rng = np.random.default_rng(9)
    a, b = rng.normal(0, 1, 50), rng.normal(1, 1, 50)
    _, p = permutation_test(a, b, n_perm=2000, alternative=alternative, rng=rng)
    assert (p < 0.01) is small
