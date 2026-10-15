"""Calificaciones sintéticas con la estructura de las reales, para notebooks y tests."""

import numpy as np
import pandas as pd


def synthetic_ratings(
    n_users: int = 800,
    n_items: int = 500,
    rank: int = 5,
    density: float = 0.05,
    noise: float = 0.5,
    popularity_skew: float = 1.0,
    mnar: float = 1.0,
    signal: float = 1.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Genera (calificaciones observadas, verdad) con:

    - gusto de bajo rango: s_ui = μ + b_u + b_i + p_uᵀ q_i, recortado a [1, 5] y redondeado;
    - popularidad de cola larga: la probabilidad de observar un ítem ∝ (rango de popularidad)^−skew;
    - faltantes no aleatorios (MNAR): con mnar > 0 se observan más los pares que gustan, como en
      la vida real (uno califica lo que eligió ver). Esto sesga las estimaciones ingenuas.

    `signal` escala la parte de bajo rango: la interacción p_uᵀq_i tiene desviación ≈ signal/√rank.
    Con signal = 1 y rank = 5 es ≈ 0.45, menor que el ruido típico; subirla da a la factorización
    más estructura que recuperar.
    """
    rng = np.random.default_rng(seed)
    P = rng.normal(0, np.sqrt(signal) / np.sqrt(rank), (n_users, rank))
    Q = rng.normal(0, np.sqrt(signal) / np.sqrt(rank), (n_items, rank))
    bu, bi = rng.normal(0, 0.3, n_users), rng.normal(0, 0.4, n_items)
    true = 3.5 + bu[:, None] + bi[None, :] + P @ Q.T
    pop = (np.arange(1, n_items + 1, dtype=float) ** -popularity_skew)[rng.permutation(n_items)]
    logits = np.log(pop)[None, :] + mnar * (true - true.mean())
    prob = np.exp(logits)
    prob *= density * n_users * n_items / prob.sum()
    observed = rng.random((n_users, n_items)) < np.clip(prob, 0, 1)
    u, i = np.nonzero(observed)
    ratings = np.clip(np.rint(true[u, i] + rng.normal(0, noise, len(u))), 1, 5)
    df = pd.DataFrame(
        {
            "u": u.astype(np.int32),
            "i": i.astype(np.int32),
            "rating": ratings,
            "ts": rng.permutation(len(u)),
        }
    )
    obs_prob = np.clip(
        prob, 0, 1
    )  # probabilidad real de observar cada par (desconocida en la vida real)
    return df, {"true": true, "P": P, "Q": Q, "popularity": pop, "obs_prob": obs_prob}
