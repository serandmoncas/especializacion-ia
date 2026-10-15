"""A/B bayesiano con el modelo conjugado Beta–Binomial."""

import numpy as np


def beta_posterior(
    successes: int, n: int, prior: tuple[float, float] = (1.0, 1.0)
) -> tuple[float, float]:
    """Beta(α, β) a priori + k éxitos en n → Beta(α + k, β + n − k) a posteriori.

    El prior Beta(α, β) equivale a haber visto α − 1 éxitos y β − 1 fracasos antes.
    """
    a, b = prior
    return a + successes, b + n - successes


def prob_b_beats_a(
    post_a: tuple[float, float],
    post_b: tuple[float, float],
    *,
    n_samples: int = 200_000,
    rng: np.random.Generator | None = None,
) -> float:
    """P(p_b > p_a | datos) por Monte Carlo sobre las posteriores."""
    rng = rng or np.random.default_rng(0)
    pa = rng.beta(*post_a, size=n_samples)
    pb = rng.beta(*post_b, size=n_samples)
    return float(np.mean(pb > pa))


def expected_loss(
    post_a: tuple[float, float],
    post_b: tuple[float, float],
    *,
    n_samples: int = 200_000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Pérdida esperada de elegir A y de elegir B: E[max(p_otro − p_elegido, 0)].

    Responde “si me equivoco, ¿cuánto pierdo en promedio?”: un criterio de decisión
    más útil que P(B > A) cuando la diferencia es pequeña y no importa.
    """
    rng = rng or np.random.default_rng(0)
    pa = rng.beta(*post_a, size=n_samples)
    pb = rng.beta(*post_b, size=n_samples)
    return float(np.mean(np.maximum(pb - pa, 0))), float(np.mean(np.maximum(pa - pb, 0)))
