"""Estadística aplicada: inferencia, remuestreo, bayesiana y diseño de experimentos desde cero."""

from applied_stats.bayes import beta_posterior, expected_loss, prob_b_beats_a
from applied_stats.experiments import (
    cuped,
    power_two_proportions,
    sample_ratio_mismatch,
    sample_size_two_proportions,
)
from applied_stats.inference import TestResult, mean_ci, two_proportion_ztest, welch_ttest
from applied_stats.multiple import benjamini_hochberg, bonferroni, holm
from applied_stats.resampling import bootstrap_ci, bootstrap_diff_ci, permutation_test

__all__ = [
    "TestResult",
    "benjamini_hochberg",
    "beta_posterior",
    "bonferroni",
    "bootstrap_ci",
    "bootstrap_diff_ci",
    "cuped",
    "expected_loss",
    "holm",
    "mean_ci",
    "permutation_test",
    "power_two_proportions",
    "prob_b_beats_a",
    "sample_ratio_mismatch",
    "sample_size_two_proportions",
    "two_proportion_ztest",
    "welch_ttest",
]
