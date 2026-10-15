import numpy as np
import pytest
from applied_stats import beta_posterior, expected_loss, prob_b_beats_a
from scipy import integrate, stats


def test_beta_posterior_conjugate_update():
    assert beta_posterior(30, 100) == (31, 71)
    assert beta_posterior(30, 100, prior=(2, 5)) == (32, 75)


def test_prob_b_beats_a_matches_numerical_integration():
    pa, pb = beta_posterior(40, 400), beta_posterior(55, 400)
    exact, _ = integrate.quad(
        lambda x: stats.beta.pdf(x, *pb) * stats.beta.cdf(x, *pa), 0, 1, limit=200
    )
    assert prob_b_beats_a(pa, pb) == pytest.approx(exact, abs=0.005)


def test_prob_is_one_half_for_identical_posteriors():
    post = beta_posterior(10, 100)
    assert prob_b_beats_a(post, post) == pytest.approx(0.5, abs=0.005)


def test_expected_loss_favors_better_arm():
    pa, pb = beta_posterior(100, 1000), beta_posterior(150, 1000)
    loss_a, loss_b = expected_loss(pa, pb, rng=np.random.default_rng(1))
    assert loss_b < 1e-4 < loss_a
    assert loss_a == pytest.approx(0.05, abs=0.005)  # ≈ p_b − p_a cuando B gana claramente
