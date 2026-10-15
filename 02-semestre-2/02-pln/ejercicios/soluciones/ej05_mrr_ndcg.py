"""Ejercicio 5: MRR y NDCG con relevancia graduada. Ver README.md."""

import numpy as np


def mrr(rankings: list[list[str]], relevant: list[set[str]]) -> float:
    """Media de 1/posición del primer relevante (0 si no aparece)."""
    out = []
    for ranked, rel in zip(rankings, relevant, strict=True):
        out.append(next((1 / (i + 1) for i, r in enumerate(ranked) if r in rel), 0.0))
    return float(np.mean(out))


def ndcg_graded(ranked: list[str], gains: dict[str, float], k: int) -> float:
    """DCG = Σ gain / log2(pos + 1); el ideal ordena las ganancias de mayor a menor."""
    dcg = sum(gains.get(r, 0.0) / np.log2(i + 2) for i, r in enumerate(ranked[:k]))
    best = sorted(gains.values(), reverse=True)[:k]
    ideal = sum(g / np.log2(i + 2) for i, g in enumerate(best))
    return float(dcg / ideal) if ideal > 0 else 0.0


def check() -> None:
    from sklearn.metrics import ndcg_score

    assert (
        abs(mrr([["a", "b"], ["x", "y", "b"], ["z"]], [{"b"}, {"b"}, {"q"}]) - (1 / 2 + 1 / 3) / 3)
        < 1e-12
    )
    rng = np.random.default_rng(0)
    for _ in range(100):
        ids = [f"d{i}" for i in range(12)]
        gains = {d: float(rng.integers(0, 4)) for d in ids}
        if not any(gains.values()):
            continue
        scores = rng.random(12)
        ranked = [ids[i] for i in np.argsort(-scores)]
        ref = ndcg_score([[gains[d] for d in ids]], [scores], k=5)
        assert abs(ndcg_graded(ranked, gains, 5) - ref) < 1e-12
    print("ej05 ok — MRR a mano y NDCG igual que sklearn")


if __name__ == "__main__":
    check()
