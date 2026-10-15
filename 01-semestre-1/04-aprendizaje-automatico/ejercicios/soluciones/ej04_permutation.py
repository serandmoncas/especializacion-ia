"""Solución ejercicio 4: importancia por permutación.

Con features correlacionadas, permutar una deja a la otra como sustituta: el modelo pierde poco
y ambas parecen poco importantes, aunque juntas sean cruciales. En Adult pasa con `sex`, cuya
información también está en `relationship` (Husband/Wife): ver el proyecto.
"""

import numpy as np


def permutation_importance(
    model, X, y, scorer, *, n_repeats: int = 5, rng=None
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (media, desviación) de la caída del puntaje por columna."""
    rng = rng or np.random.default_rng(0)
    X = np.asarray(X, dtype=float)
    base = scorer(model, X, y)
    drops = np.zeros((X.shape[1], n_repeats))
    for j in range(X.shape[1]):
        for r in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops[j, r] = base - scorer(model, Xp, y)
    return drops.mean(axis=1), drops.std(axis=1)


def check() -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance as sk_perm

    rng = np.random.default_rng(0)
    n = 3000
    signal = rng.normal(size=n)
    X = np.column_stack(
        [
            signal,  # 0: informativa
            signal + rng.normal(scale=0.05, size=n),  # 1: copia casi exacta de la 0
            rng.normal(size=n),  # 2: algo informativa
            rng.normal(size=n),  # 3: ruido
        ]
    )
    y = (signal + 0.5 * X[:, 2] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    model = RandomForestClassifier(100, random_state=0, n_jobs=-1).fit(X[:2000], y[:2000])

    def acc(m, X_, y_):
        return m.score(X_, y_)

    ours, _ = permutation_importance(model, X[2000:], y[2000:], acc, n_repeats=10)
    ref = sk_perm(model, X[2000:], y[2000:], n_repeats=10, random_state=0).importances_mean
    np.testing.assert_allclose(ours, ref, atol=0.02)
    assert ours[3] < 0.01, "el ruido no debería importar"
    solo = RandomForestClassifier(100, random_state=0, n_jobs=-1).fit(X[:2000, [0, 2, 3]], y[:2000])
    ours_solo, _ = permutation_importance(solo, X[2000:, [0, 2, 3]], y[2000:], acc, n_repeats=10)
    assert ours_solo[0] > ours[0] + ours[1], "sin la copia, la feature 0 debería importar más"
    print(f"ej04 ok — con copia: {ours.round(3)} | sin copia, feature 0: {ours_solo[0]:.3f}")


if __name__ == "__main__":
    check()
