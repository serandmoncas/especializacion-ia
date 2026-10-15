"""Solución ejercicio 4: Thompson sampling vs. reparto 50/50.

Thompson juega cada brazo con probabilidad igual a la probabilidad posterior de que sea el
mejor: explora mientras hay incertidumbre y explota cuando se resuelve. El regret crece como
O(log T) frente a O(T) del reparto fijo. Lo que se pierde: el brazo malo recibe pocas
observaciones, así que su tasa queda mal estimada y el análisis inferencial clásico
(p-valores con n fijo) deja de ser válido.
"""

import numpy as np


def thompson(probs: list[float], rounds: int, seed: int = 0) -> tuple[np.ndarray, float]:
    """Devuelve (veces que se jugó cada brazo, regret acumulado esperado)."""
    rng = np.random.default_rng(seed)
    k = len(probs)
    alpha, beta = np.ones(k), np.ones(k)
    pulls = np.zeros(k, dtype=int)
    for _ in range(rounds):
        arm = int(np.argmax(rng.beta(alpha, beta)))
        reward = rng.random() < probs[arm]
        alpha[arm] += reward
        beta[arm] += 1 - reward
        pulls[arm] += 1
    regret = float(np.sum(pulls * (max(probs) - np.asarray(probs))))
    return pulls, regret


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
