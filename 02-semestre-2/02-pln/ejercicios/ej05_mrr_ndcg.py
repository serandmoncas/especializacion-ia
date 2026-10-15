"""Ejercicio 5: MRR y NDCG con relevancia graduada. Ver README.md."""

import numpy as np


def mrr(rankings: list[list[str]], relevant: list[set[str]]) -> float:
    """Media de 1/posición del primer relevante (0 si no aparece)."""
    raise NotImplementedError


def ndcg_graded(ranked: list[str], gains: dict[str, float], k: int) -> float:
    """DCG = Σ gain / log2(pos + 1); el ideal ordena las ganancias de mayor a menor."""
    raise NotImplementedError


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
