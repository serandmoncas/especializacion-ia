"""Ejercicio 3: diversificación con MMR. Ver README.md."""

import numpy as np


def mmr(relevance: np.ndarray, sim: np.ndarray, k: int, lam: float, pool: int = 100) -> list[int]:
    raise NotImplementedError


def intra_list_similarity(items: list[int], sim: np.ndarray) -> float:
    s = sim[np.ix_(items, items)]
    n = len(items)
    return float((s.sum() - np.trace(s)) / (n * (n - 1)))


def check() -> None:
    from applied_recsys.data import to_csr
    from applied_recsys.factorization import ImplicitALS
    from applied_recsys.neighbors import item_similarity
    from applied_recsys.synthetic import synthetic_ratings

    df, _ = synthetic_ratings(800, 500, density=0.05, seed=4)
    R = to_csr(df, (800, 500))
    sim = item_similarity(R)
    model = ImplicitALS(k=32, alpha=1.0, n_iter=8).fit(R)
    scores = model.scores(np.arange(50))
    rows = []
    for lam in (1.0, 0.7, 0.4):
        rel_sum, ils = [], []
        for u in range(50):
            s = scores[u].copy()
            s[R[u].indices] = -np.inf
            items = mmr(s, sim, 10, lam)
            rel_sum.append(scores[u, items].sum())
            ils.append(intra_list_similarity(items, sim))
        rows.append((lam, np.mean(rel_sum), np.mean(ils)))
    rels, ilss = [r[1] for r in rows], [r[2] for r in rows]
    assert rels[0] >= rels[1] >= rels[2] and ilss[0] > ilss[1] > ilss[2], rows
    print(
        "ej03 ok — (λ, relevancia, similitud intra-lista):",
        [(lam, round(r, 2), round(s, 3)) for lam, r, s in rows],
    )


if __name__ == "__main__":
    check()
