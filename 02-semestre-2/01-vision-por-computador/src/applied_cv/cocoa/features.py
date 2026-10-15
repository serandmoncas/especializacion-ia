"""Rasgos hechos a mano para el baseline «color + textura»."""

import numpy as np

from applied_cv.image import color_histogram, lbp_histogram, rgb_to_gray


def color_texture_features(rgb: np.ndarray, bins: int = 16) -> np.ndarray:
    """Histogramas HSV (3·bins) + histograma LBP (256) de un recorte RGB uint8."""
    x = np.asarray(rgb, dtype=float) / 255.0
    return np.concatenate([color_histogram(x, bins), lbp_histogram(np.round(rgb_to_gray(x) * 255))])
