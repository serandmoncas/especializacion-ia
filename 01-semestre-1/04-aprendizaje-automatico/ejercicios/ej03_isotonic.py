"""Ejercicio 3: regresión isotónica (PAV) para calibrar. Ver README.md."""

import numpy as np


def pav(y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Ajuste isotónico de y (ya ordenado por la variable explicativa)."""
    raise NotImplementedError


class IsotonicCalibrator:
    """Mapea puntajes a probabilidades calibradas, interpolando linealmente entre bloques."""

    def fit(self, scores, y):
        raise NotImplementedError

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
