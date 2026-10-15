import numpy as np
import pandas as pd
import pytest
from applied_ts.evaluation import backtest, mae, mase, rmse, rolling_origin, smape
from applied_ts.features import calendar_features, lag_features
from applied_ts.models import naive, seasonal_naive


def test_metrics():
    y, yhat = np.array([1.0, 2, 3]), np.array([1.0, 3, 5])
    assert mae(y, yhat) == pytest.approx(1.0)
    assert rmse(y, yhat) == pytest.approx(np.sqrt(5 / 3))
    assert smape([0, 2], [0, 2]) == 0.0


def test_mase_of_seasonal_naive_in_sample_is_one():
    rng = np.random.default_rng(0)
    y = np.tile(np.arange(7.0), 30) + rng.normal(0, 0.5, 210)
    insample_pred = y[:-7]
    assert mase(y[7:], insample_pred, y, m=7) == pytest.approx(1.0)


def test_rolling_origin_never_leaks_future():
    splits = list(rolling_origin(20, initial=10, horizon=3, step=3))
    assert len(splits) == 3
    for train, test in splits:
        assert train.max() < test.min()
        assert len(test) == 3


def test_backtest_shapes_and_seasonal_naive_wins_on_seasonal_data():
    y = np.tile([0.0, 5, 10, 5], 50)
    nv = backtest(y, naive, initial=100, horizon=4)
    sn = backtest(y, lambda h, k: seasonal_naive(h, k, 4), initial=100, horizon=4)
    assert nv["pred"].shape == (25, 4)
    assert sn["abs_err_by_step"].max() == 0.0 < nv["abs_err_by_step"].mean()


def test_lag_features_use_only_past():
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    y = pd.Series(np.arange(10.0), index=idx)
    f = lag_features(y, [2, 3], rolling=[2], min_lag=2)
    assert f.loc[idx[5], "lag_2"] == 3.0
    assert f.loc[idx[5], "roll_mean_2"] == pytest.approx((2 + 3) / 2)  # termina en t − 2
    with pytest.raises(ValueError):
        lag_features(y, [1], min_lag=24)


def test_calendar_features_cyclic_hour():
    idx = pd.date_range("2024-01-01 22:00", periods=4, freq="h")
    cal = calendar_features(idx)
    d_23_0 = np.hypot(
        cal.hour_sin.iloc[1] - cal.hour_sin.iloc[2], cal.hour_cos.iloc[1] - cal.hour_cos.iloc[2]
    )
    d_22_23 = np.hypot(
        cal.hour_sin.iloc[0] - cal.hour_sin.iloc[1], cal.hour_cos.iloc[0] - cal.hour_cos.iloc[1]
    )
    assert d_23_0 == pytest.approx(d_22_23)  # 23→0 está tan cerca como 22→23
