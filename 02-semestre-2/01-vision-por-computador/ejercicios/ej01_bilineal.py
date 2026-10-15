"""Ejercicio 1: redimensionar con interpolación bilineal. Ver README.md."""

import numpy as np


def resize_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Imagen gris [H, W] → [out_h, out_w], bilineal con align_corners=False."""
    raise NotImplementedError


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
