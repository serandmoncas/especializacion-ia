"""Ejercicio 6: HSV → RGB, la inversa de `applied_cv.image.rgb_to_hsv`. Ver README.md."""

import numpy as np


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """HSV en [0, 1] → RGB en [0, 1]."""
    raise NotImplementedError


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
