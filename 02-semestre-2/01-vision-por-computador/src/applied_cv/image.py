"""Operaciones clásicas sobre imágenes, desde cero con NumPy."""

import numpy as np

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
# Pesos de luminancia ITU-R 709, los mismos que skimage.color.rgb2gray.
GRAY_WEIGHTS = np.array([0.2125, 0.7154, 0.0721])


def conv2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Correlación 2D del mismo tamaño, con relleno por reflexión de borde.

    Equivale a `scipy.ndimage.correlate(mode="reflect")` (que NumPy llama "symmetric"). Las CNN
    también calculan correlaciones, aunque se las llame convoluciones.
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(
        np.asarray(image, dtype=float), ((ph, kh - 1 - ph), (pw, kw - 1 - pw)), mode="symmetric"
    )
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel)


def sobel(image: np.ndarray) -> np.ndarray:
    """Magnitud del gradiente con los núcleos de Sobel (bordes)."""
    return np.hypot(conv2d(image, SOBEL_X), conv2d(image, SOBEL_X.T))


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB en [0, 1] → HSV en [0, 1] (mismo convenio que skimage.color.rgb2hsv).

    El tono (H) separa «qué color» de «cuánta luz» (V): por eso el café de una lesión se reconoce
    mejor en H que en R, G y B, que cambian todos con la iluminación.
    """
    rgb = np.asarray(rgb, dtype=float)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    v = rgb.max(axis=-1)
    delta = v - rgb.min(axis=-1)
    s = np.divide(delta, v, out=np.zeros_like(v), where=v > 0)
    safe = np.where(delta > 0, delta, 1.0)
    h = np.select(
        [delta == 0, v == r, v == g],
        [0.0, ((g - b) / safe) % 6, (b - r) / safe + 2],
        (r - g) / safe + 4,
    )
    return np.stack([h / 6, s, v], axis=-1)


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Luminancia en [0, 1] a partir de RGB en [0, 1]."""
    return np.asarray(rgb, dtype=float)[..., :3] @ GRAY_WEIGHTS


def local_binary_pattern(gray: np.ndarray) -> np.ndarray:
    """LBP de 8 vecinos en radio 1 con interpolación bilineal (skimage, method="default").

    Cada vecino sobre el círculo aporta un bit: 1 si es ≥ que el pixel central. El código de 8 bits
    describe la micro-textura local (borde, esquina, punto, zona plana) sin depender del brillo.
    Fuera de la imagen el valor es 0, como en skimage.
    """
    g = np.asarray(gray, dtype=float)
    h, w = g.shape
    padded = np.pad(g, 1, mode="constant")
    out = np.zeros((h, w), dtype=np.int64)
    rows, cols = np.mgrid[0:h, 0:w]
    for p in range(8):
        angle = 2 * np.pi * p / 8
        dr, dc = round(-np.sin(angle), 5), round(np.cos(angle), 5)
        r, c = rows + 1 + dr, cols + 1 + dc
        r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
        fr, fc = r - r0, c - c0
        r1, c1 = np.minimum(r0 + 1, h + 1), np.minimum(c0 + 1, w + 1)
        value = (
            padded[r0, c0] * (1 - fr) * (1 - fc)
            + padded[r0, c1] * (1 - fr) * fc
            + padded[r1, c0] * fr * (1 - fc)
            + padded[r1, c1] * fr * fc
        )
        out |= (value >= g).astype(np.int64) << p
    return out


def color_histogram(rgb: np.ndarray, bins: int = 16) -> np.ndarray:
    """Histogramas normalizados de H, S y V concatenados (3·bins valores)."""
    hsv = rgb_to_hsv(rgb).reshape(-1, 3)
    return np.concatenate(
        [np.histogram(hsv[:, i], bins=bins, range=(0.0, 1.0))[0] / len(hsv) for i in range(3)]
    )


def lbp_histogram(gray255: np.ndarray) -> np.ndarray:
    """Histograma normalizado de los 256 códigos LBP de una imagen gris en [0, 255]."""
    codes = local_binary_pattern(gray255)
    return np.bincount(codes.ravel(), minlength=256) / codes.size
