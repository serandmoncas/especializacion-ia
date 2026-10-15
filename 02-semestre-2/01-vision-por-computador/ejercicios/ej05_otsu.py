"""Ejercicio 5: umbral de Otsu (1979). Ver README.md."""

import numpy as np


def otsu_threshold(gray: np.ndarray) -> int:
    """Umbral de Otsu de una imagen gris entera: los pixeles > t son primer plano."""
    raise NotImplementedError


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
