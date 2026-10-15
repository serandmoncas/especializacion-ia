"""Solución ejercicio 4: IPS para corregir la evaluación offline.

Estimador ingenuo de la precisión@10: fracción de la lista que aparece como relevante en el test.
Subestima, y mucho más para modelos que recomiendan lo poco expuesto: un ítem de nicho rara vez
está en el test aunque al usuario le encantaría.

IPS: cada acierto observado vale 1/p_test(u, i). En esperanza, Σ_i 1[obs, rel]/p = Σ_i P(rel): la
precisión real. Costo: varianza. Si p es diminuta, un solo acierto pesa enormemente (puede dar una
“precisión” > 1). Y en la vida real p no se conoce: estimarla con la popularidad no captura que se
observa más lo que gusta (selección por el propio gusto), y el sesgo vuelve.
"""

import numpy as np
from scipy.stats import norm, spearmanr


def precision_estimates(
    recs: np.ndarray, users: np.ndarray, relevant_test: np.ndarray, p_test: np.ndarray
) -> tuple[float, float]:
    """(ingenua, IPS) de la precisión@k. relevant_test[u, i]: ¿relevante y observado en test?"""
    rows = users[:, None]
    hits = relevant_test[rows, recs]
    return float(hits.mean()), float((hits / p_test[rows, recs]).mean())


def check() -> None:
    from applied_recsys.baselines import BiasModel, popularity_scores
    from applied_recsys.data import random_split, to_csr
    from applied_recsys.factorization import FunkSVD, ImplicitALS, PureSVD
    from applied_recsys.metrics import top_n
    from applied_recsys.neighbors import ItemKNN
    from applied_recsys.synthetic import synthetic_ratings

    shape = (1500, 800)
    df, truth = synthetic_ratings(*shape, density=0.04, mnar=1.0, noise=0.5, seed=2)
    tr, te = random_split(df, 0.25, seed=0)
    R = to_csr(tr, shape)
    pop = popularity_scores(R)
    funk = FunkSVD(k=10, epochs=30).fit(R)
    models = {
        "popularidad": lambda us: np.tile(pop, (len(us), 1)),
        "FunkSVD": lambda us: (
            funk.mu + funk.bu[us, None] + funk.bi[None, :] + funk.P[us] @ funk.Q.T
        ),
        "item-kNN": ItemKNN(k=50).fit(R, BiasModel().fit(R)).scores,
        "PureSVD": PureSVD(20).fit(R).scores,
        "ALS implícito": ImplicitALS(k=32, alpha=1.0, n_iter=10).fit(R).scores,
    }
    relevant_test = np.zeros(shape, dtype=bool)
    pos = te[te.rating >= 4]
    relevant_test[pos.u, pos.i] = True
    p = truth["obs_prob"]
    p_test = 0.25 * p / (1 - 0.75 * p)  # P(en test | no está en train)
    p_rel = norm.sf((3.5 - truth["true"]) / 0.5)  # P(calificación ≥ 4 | gusto real)
    users = np.arange(shape[0])
    rows = []
    for name, fn in models.items():
        recs = top_n(fn(users), R[users], 10)
        naive, ips = precision_estimates(recs, users, relevant_test, p_test)
        rows.append((name, p_rel[users[:, None], recs].mean(), naive, ips))
    real, naive, ips = (np.array([r[k] for r in rows]) for k in (1, 2, 3))
    assert spearmanr(naive, real)[0] < 0 < 0.8 <= spearmanr(ips, real)[0], rows
    assert np.median(np.abs(ips - real)) < 0.1 < np.median(np.abs(naive - real))
    print("ej04 ok — precisión@10 (real | ingenua | IPS):")
    for name, r, n, i in rows:
        print(f"  {name:<14} {r:.3f} | {n:.3f} | {i:.3f}")


if __name__ == "__main__":
    check()
