"""Ejercicio 4: Thompson sampling vs. reparto 50/50. Ver README.md."""

import numpy as np


def thompson(probs: list[float], rounds: int, seed: int = 0) -> tuple[np.ndarray, float]:
    """Devuelve (veces que se jugó cada brazo, regret acumulado esperado)."""
    raise NotImplementedError


def uniform_regret(probs: list[float], rounds: int) -> float:
    return float(rounds / len(probs) * np.sum(max(probs) - np.asarray(probs)))


def check() -> None:
    probs, rounds = [0.10, 0.12], 20_000
    pulls, regret = thompson(probs, rounds)
    base = uniform_regret(probs, rounds)
    assert pulls[1] / rounds > 0.7, f"el mejor brazo solo recibió {pulls[1] / rounds:.0%}"
    assert regret < 0.5 * base, (regret, base)
    print(
        f"ej04 ok — tráfico al mejor brazo {pulls[1] / rounds:.0%}; "
        f"regret Thompson {regret:.0f} vs 50/50 {base:.0f} conversiones"
    )


if __name__ == "__main__":
    check()
