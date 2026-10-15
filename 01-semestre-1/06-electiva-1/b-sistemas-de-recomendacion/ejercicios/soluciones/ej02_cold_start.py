"""Solución ejercicio 2: arranque en frío por atributos.

Con los ítems conocidos se ajusta una regresión ridge atributos → (b_i, q_i). Para un ítem nuevo se
predicen sus factores y su sesgo y se usa la misma fórmula del modelo. Funciona en la medida en que
los atributos expliquen el gusto: si dos películas del mismo género gustan a públicos distintos, el
modelo frío no lo puede saber.
"""

import numpy as np


def fit_feature_map(features: np.ndarray, targets: np.ndarray, reg: float = 1.0) -> np.ndarray:
    """Ridge con intercepto: W tal que [1, x] W ≈ objetivo (solución cerrada)."""
    X = np.column_stack([np.ones(len(features)), features])
    penalty = reg * np.eye(X.shape[1])
    penalty[0, 0] = 0.0  # no penalizar el intercepto
    return np.linalg.solve(X.T @ X + penalty, X.T @ targets)


def predict_cold(features: np.ndarray, W: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(features)), features]) @ W


def check() -> None:
    from applied_recsys.baselines import BiasModel
    from applied_recsys.data import to_csr
    from applied_recsys.factorization import ExplicitALS
    from applied_recsys.synthetic import synthetic_ratings

    rng = np.random.default_rng(0)
    n_u, n_i = 1200, 700
    df, truth = synthetic_ratings(n_u, n_i, density=0.06, rank=5, signal=4.0, mnar=0.0, seed=3)
    # atributos ruidosos que reflejan los factores reales (como un género o un director)
    features = truth["Q"] @ rng.normal(size=(5, 12)) + rng.normal(0, 0.3, (n_i, 12))
    cold = rng.choice(n_i, 100, replace=False)
    is_cold = np.isin(df.i, cold)
    warm_df, cold_df = df[~is_cold], df[is_cold]
    R = to_csr(warm_df, (n_u, n_i))
    bias = BiasModel().fit(R)
    als = ExplicitALS(k=5, reg=0.1, n_iter=15).fit(R, bias)
    warm_items = np.setdiff1d(np.unique(warm_df.i), cold)
    W = fit_feature_map(
        features[warm_items], np.column_stack([bias.bi[warm_items], als.Q[warm_items]])
    )
    pred = predict_cold(features[cold], W)
    bi_cold, Q_cold = np.zeros(n_i), np.zeros((n_i, 5))
    bi_cold[cold], Q_cold[cold] = pred[:, 0], pred[:, 1:]
    u, i, y = cold_df.u.to_numpy(), cold_df.i.to_numpy(), cold_df.rating.to_numpy()
    rmse_feat = np.sqrt(
        np.mean(
            (
                y
                - np.clip(
                    bias.mu + bias.bu[u] + bi_cold[i] + np.einsum("nk,nk->n", als.P[u], Q_cold[i]),
                    1,
                    5,
                )
            )
            ** 2
        )
    )
    rmse_base = np.sqrt(np.mean((y - np.clip(bias.mu + bias.bu[u], 1, 5)) ** 2))
    assert rmse_feat < 0.85 * rmse_base, (rmse_feat, rmse_base)
    print(
        f"ej02 ok — RMSE en ítems fríos: con atributos {rmse_feat:.3f}"
        f" | solo μ + b_u {rmse_base:.3f}"
    )


if __name__ == "__main__":
    check()
