"""Solución ejercicio 5: umbral de Otsu (1979).

Para cada umbral t se separan los pixeles en ≤ t y > t; Otsu elige el t que maximiza la varianza
entre las dos clases, w0·w1·(μ0 − μ1)², calculable para todos los t con sumas acumuladas.
"""

import numpy as np


def otsu_threshold(gray: np.ndarray) -> int:
    g = np.asarray(gray).astype(int)
    values = np.arange(g.min(), g.max() + 1)
    p = np.bincount((g - g.min()).ravel(), minlength=len(values)) / g.size
    w0 = np.cumsum(p)
    mu = np.cumsum(p * values)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * w0 - mu) ** 2 / (w0 * (1 - w0))
    between = np.nan_to_num(between[:-1], nan=-1.0)
    return int(values[np.argmax(between)])


def check() -> None:
    from skimage.filters import threshold_otsu

    for seed in range(5):
        rng = np.random.default_rng(seed)
        pixels = np.r_[rng.normal(60, 15, 3000), rng.normal(170, 25, 2000)]
        gray = pixels.clip(0, 255).astype(np.uint8).reshape(50, 100)
        assert otsu_threshold(gray) == threshold_otsu(gray), seed
    print("ej05 ok — coincide con skimage.filters.threshold_otsu")


if __name__ == "__main__":
    check()
