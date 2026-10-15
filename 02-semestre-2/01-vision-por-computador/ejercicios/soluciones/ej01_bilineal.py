"""Solución ejercicio 1: redimensionar con interpolación bilineal.

Cada pixel de salida mira el punto (i + 0.5)·escala − 0.5 de la entrada (centros de pixel
alineados, como `align_corners=False`), recortado a la imagen, y mezcla sus cuatro vecinos con
pesos proporcionales a la cercanía. Primero se interpola en columnas y luego en filas.
"""

import numpy as np


def resize_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = img.shape

    def coords(n_out: int, n_in: int):
        src = np.clip((np.arange(n_out) + 0.5) * n_in / n_out - 0.5, 0, n_in - 1)
        lo = np.floor(src).astype(int)
        return lo, np.minimum(lo + 1, n_in - 1), src - lo

    r0, r1, fr = coords(out_h, h)
    c0, c1, fc = coords(out_w, w)
    top = img[r0][:, c0] * (1 - fc) + img[r0][:, c1] * fc
    bottom = img[r1][:, c0] * (1 - fc) + img[r1][:, c1] * fc
    return top * (1 - fr)[:, None] + bottom * fr[:, None]


def check() -> None:
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(0)
    for h, w, oh, ow in [(5, 7, 11, 13), (8, 8, 3, 5), (6, 4, 6, 4), (4, 9, 10, 2)]:
        x = rng.random((h, w))
        ref = F.interpolate(
            torch.tensor(x)[None, None], size=(oh, ow), mode="bilinear", align_corners=False
        )[0, 0].numpy()
        np.testing.assert_allclose(resize_bilinear(x, oh, ow), ref, atol=1e-12)
    print("ej01 ok — coincide con torch.nn.functional.interpolate")


if __name__ == "__main__":
    check()
