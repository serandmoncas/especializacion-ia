"""Solución ejercicio 3: regresión isotónica por Pool Adjacent Violators (PAV).

Se mantiene una pila de bloques (suma, conteo). Cada punto nuevo entra como bloque propio; mientras
la media del último bloque sea menor que la del anterior (violación de monotonía), se fusionan.
O(n) amortizado tras ordenar. La solución es la proyección de y sobre el cono de vectores
no decrecientes: el mínimo de Σ(y − g)² con g monótona.
"""

import numpy as np


def pav(y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Ajuste isotónico de y (ya ordenado por la variable explicativa)."""
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    sums, counts, sizes = [], [], []
    for yi, wi in zip(y, w, strict=True):
        sums.append(yi * wi)
        counts.append(wi)
        sizes.append(1)
        while len(sums) > 1 and sums[-2] / counts[-2] > sums[-1] / counts[-1]:
            s, c, n = sums.pop(), counts.pop(), sizes.pop()
            sums[-1] += s
            counts[-1] += c
            sizes[-1] += n
    return np.repeat([s / c for s, c in zip(sums, counts, strict=True)], sizes)


class IsotonicCalibrator:
    """Mapea puntajes a probabilidades calibradas, interpolando linealmente entre bloques."""

    def fit(self, scores, y):
        order = np.argsort(scores, kind="stable")
        self.x_ = np.asarray(scores, dtype=float)[order]
        self.y_ = pav(np.asarray(y, dtype=float)[order])
        return self

    def predict(self, scores) -> np.ndarray:
        return np.interp(scores, self.x_, self.y_)


def check() -> None:
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(0)
    s = np.sort(rng.random(20_000))
    y = (rng.random(20_000) < s**2).astype(float)  # puntajes descalibrados: P(y=1|s) = s²
    fitted = pav(y)
    ref = IsotonicRegression().fit_transform(s, y)
    np.testing.assert_allclose(fitted, ref, atol=1e-12)
    assert np.all(np.diff(fitted) >= -1e-12), "la salida debe ser no decreciente"
    cal = IsotonicCalibrator().fit(s, y)
    grid = np.array([0.2, 0.5, 0.8])
    print(f"ej03 ok — calibrado en {grid}: {cal.predict(grid).round(2)} (verdad: {grid**2})")


if __name__ == "__main__":
    check()
