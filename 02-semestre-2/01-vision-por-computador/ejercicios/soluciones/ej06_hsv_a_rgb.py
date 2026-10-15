"""Solución ejercicio 6: HSV → RGB, la inversa de `applied_cv.image.rgb_to_hsv`.

El tono se parte en 6 sectores de 60°; en cada uno un canal vale V, otro p = V(1 − S) y el tercero
sube (t) o baja (q) linealmente con la fracción f del sector.
"""

import numpy as np


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    hsv = np.asarray(hsv, dtype=float)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h6 = (h % 1.0) * 6
    sector = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    table = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    channels = [np.choose(sector, [row[c] for row in table]) for c in range(3)]
    return np.stack(channels, axis=-1)


def check() -> None:
    from applied_cv.image import rgb_to_hsv
    from skimage.color import hsv2rgb

    rng = np.random.default_rng(0)
    rgb = rng.random((30, 30, 3))
    np.testing.assert_allclose(hsv_to_rgb(rgb_to_hsv(rgb)), rgb, atol=1e-12)
    hsv = rng.random((30, 30, 3))
    np.testing.assert_allclose(hsv_to_rgb(hsv), hsv2rgb(hsv), atol=1e-12)
    print("ej06 ok — ida y vuelta exacta y coincide con skimage.color.hsv2rgb")


if __name__ == "__main__":
    check()
