import numpy as np
import pytest
from applied_ts.autocorrelation import acf, difference, pacf, undifference
from applied_ts.decomposition import centered_moving_average, classical_decompose
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf as sm_acf
from statsmodels.tsa.stattools import pacf as sm_pacf


@pytest.fixture
def seasonal_series():
    rng = np.random.default_rng(0)
    t = np.arange(240)
    return 10 + 0.05 * t + 3 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, len(t))


@pytest.mark.parametrize("period", [12, 7])
def test_decomposition_matches_statsmodels(seasonal_series, period):
    ours = classical_decompose(seasonal_series, period)
    ref = seasonal_decompose(seasonal_series, period=period, model="additive")
    np.testing.assert_allclose(ours["trend"], ref.trend, equal_nan=True)
    np.testing.assert_allclose(ours["seasonal"], ref.seasonal, atol=1e-10)


def test_decomposition_recovers_components(seasonal_series):
    dec = classical_decompose(seasonal_series, 12)
    truth = 3 * np.sin(2 * np.pi * np.arange(12) / 12)
    # Cada índice promedia ~19 ciclos con σ = 0.5: EE ≈ 0.5/√19 ≈ 0.115. Tolerancia: 4 EE.
    np.testing.assert_allclose(dec["seasonal"][:12], truth, atol=0.45)
    assert np.nanstd(dec["resid"]) == pytest.approx(0.5, abs=0.1)


def test_even_window_is_centered():
    y = np.arange(20, dtype=float)  # lineal: la media centrada debe reproducirla exactamente
    trend = centered_moving_average(y, 4)
    np.testing.assert_allclose(trend[2:-2], y[2:-2])


def test_acf_pacf_match_statsmodels(seasonal_series):
    np.testing.assert_allclose(
        acf(seasonal_series, 30), sm_acf(seasonal_series, nlags=30, fft=False)
    )
    np.testing.assert_allclose(
        pacf(seasonal_series, 20), sm_pacf(seasonal_series, nlags=20, method="ldb"), atol=1e-10
    )


def test_pacf_cuts_off_after_p_for_ar2():
    rng = np.random.default_rng(1)
    y = np.zeros(20_000)
    for t in range(2, len(y)):
        y[t] = 0.6 * y[t - 1] - 0.3 * y[t - 2] + rng.normal()
    p = pacf(y, 6)
    assert p[2] == pytest.approx(-0.3, abs=0.03)
    assert np.all(np.abs(p[3:]) < 0.03)


def test_difference_roundtrip():
    rng = np.random.default_rng(2)
    y = np.cumsum(rng.normal(size=100)) + np.tile(np.arange(7.0), 15)[:100]
    for lag in (1, 7):
        d = difference(y, lag)
        np.testing.assert_allclose(undifference(d[-10:], y[:-10], lag), y[-10:])
