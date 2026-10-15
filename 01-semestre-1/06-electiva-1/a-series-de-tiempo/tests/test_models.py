import numpy as np
import pytest
from applied_ts.models import drift, fit_ar, holt_winters, naive, seasonal_naive
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def test_baselines():
    y = np.array([1.0, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(naive(y, 2), [6, 6])
    np.testing.assert_array_equal(seasonal_naive(y, 4, 3), [4, 5, 6, 4])
    np.testing.assert_allclose(drift(y, 2), [7, 8])


def test_ar_matches_statsmodels_autoreg():
    rng = np.random.default_rng(0)
    y = np.zeros(2000)
    for t in range(2, len(y)):
        y[t] = 1.0 + 0.5 * y[t - 1] + 0.2 * y[t - 2] + rng.normal()
    ours = fit_ar(y, 2)
    ref = AutoReg(y, lags=2).fit()
    np.testing.assert_allclose([ours.intercept, *ours.coefs], ref.params, rtol=1e-10)
    np.testing.assert_allclose(
        ours.forecast(y, 5), ref.predict(start=len(y), end=len(y) + 4), rtol=1e-10
    )


def test_ar_forecast_converges_to_mean():
    fit = fit_ar(np.r_[np.zeros(5), np.ones(5)] + np.random.default_rng(1).normal(0, 0.1, 10), 1)
    long = fit.forecast(np.array([5.0]), 500)
    expected_mean = fit.intercept / (1 - fit.coefs[0])
    assert long[-1] == pytest.approx(expected_mean, rel=1e-3)


def test_holt_winters_competitive_with_statsmodels():
    rng = np.random.default_rng(3)
    t = np.arange(144)
    y = 50 + 0.4 * t + 8 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, len(t))
    train, test = y[:120], y[120:]
    ours = holt_winters(train, trend=True, seasonal_period=12)
    ref = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
    mae_ours = np.mean(np.abs(ours.forecast(24) - test))
    mae_ref = np.mean(np.abs(ref.forecast(24) - test))
    assert mae_ours < 1.3 * mae_ref + 0.2
    assert mae_ours < 2.0  # ruido σ = 1


def test_simple_exponential_smoothing_on_level_shift():
    y = np.r_[np.full(50, 10.0), np.full(50, 20.0)]
    fit = holt_winters(y, trend=False)
    assert fit.alpha > 0.9  # sin ruido, lo óptimo es seguir el último valor
    forecast = fit.forecast(3)
    assert forecast.shape == (3,)
    np.testing.assert_allclose(forecast, 20, atol=0.1)


def test_needs_two_seasons():
    with pytest.raises(ValueError):
        holt_winters(np.arange(20.0), seasonal_period=12)
