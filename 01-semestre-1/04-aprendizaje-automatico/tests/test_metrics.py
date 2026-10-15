import numpy as np
import pytest
from applied_ml.metrics import (
    average_precision,
    brier_score,
    calibration_bins,
    confusion_counts,
    expected_calibration_error,
    log_loss,
    precision_recall_f1,
    roc_auc,
    roc_curve,
)
from sklearn import metrics as skm


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.2).astype(int)
    scores = np.round(rng.normal(y * 1.2, 1.0), 1)  # redondeo: fuerza empates
    proba = 1 / (1 + np.exp(-(scores - 1)))
    return y, scores, proba


def test_confusion_and_prf_match_sklearn(scored):
    y, scores, _ = scored
    pred = scores > 0.6
    tn, fp, fn, tp = skm.confusion_matrix(y, pred).ravel()
    assert confusion_counts(y, pred) == {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    p, r, f = precision_recall_f1(y, pred)
    assert (p, r, f) == pytest.approx(
        (skm.precision_score(y, pred), skm.recall_score(y, pred), skm.f1_score(y, pred))
    )


def test_auc_matches_sklearn_with_ties(scored):
    y, scores, _ = scored
    assert roc_auc(y, scores) == pytest.approx(skm.roc_auc_score(y, scores))


def test_roc_curve_area_equals_rank_auc(scored):
    y, scores, _ = scored
    fpr, tpr, _ = roc_curve(y, scores)
    assert np.trapezoid(tpr, fpr) == pytest.approx(roc_auc(y, scores))
    assert fpr[0] == tpr[0] == 0 and fpr[-1] == tpr[-1] == 1


def test_average_precision_matches_sklearn(scored):
    y, scores, _ = scored
    assert average_precision(y, scores) == pytest.approx(skm.average_precision_score(y, scores))


def test_probabilistic_metrics_match_sklearn(scored):
    y, _, proba = scored
    assert log_loss(y, proba) == pytest.approx(skm.log_loss(y, proba))
    assert brier_score(y, proba) == pytest.approx(skm.brier_score_loss(y, proba))


def test_calibration_of_perfectly_calibrated_model():
    rng = np.random.default_rng(1)
    p = rng.random(200_000)
    y = (rng.random(200_000) < p).astype(int)
    mean_p, freq, _ = calibration_bins(y, p)
    np.testing.assert_allclose(mean_p, freq, atol=0.01)
    assert expected_calibration_error(y, p) < 0.01
    assert expected_calibration_error(y, p**3) > 0.1  # descalibrado a propósito
