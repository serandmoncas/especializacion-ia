"""Ejercicio 3: Soft-NMS gaussiano (Bodla et al., 2017). Ver README.md."""

import numpy as np
from applied_cv.boxes import box_iou  # noqa: F401 (te sirve para la solución)


def soft_nms(
    boxes: np.ndarray, scores: np.ndarray, sigma: float = 0.5, score_threshold: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    """Soft-NMS gaussiano: (índices conservados en orden, puntajes finales)."""
    raise NotImplementedError


def check() -> None:
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 20], [50, 50, 60, 60]])  # IoU(A, B) = 0.5
    keep, new_scores = soft_nms(boxes, np.array([0.9, 0.8, 0.7]), sigma=0.5)
    np.testing.assert_array_equal(keep, [0, 2, 1])
    np.testing.assert_allclose(new_scores, [0.9, 0.7, 0.8 * np.exp(-0.25 / 0.5)])
    keep, _ = soft_nms(boxes, np.array([0.9, 0.8, 0.7]), sigma=0.5, score_threshold=0.6)
    np.testing.assert_array_equal(keep, [0, 2])
    print("ej03 ok — B baja de 0.80 a 0.49 y queda detrás de C, pero no desaparece")


if __name__ == "__main__":
    check()
