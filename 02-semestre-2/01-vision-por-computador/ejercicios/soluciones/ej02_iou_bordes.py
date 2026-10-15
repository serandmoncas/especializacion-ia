"""Solución ejercicio 2: IoU de dos cajas sin errores en los casos borde.

La intersección se recorta a 0 en cada eje (cajas separadas o que solo se tocan no se intersectan) y
si la unión es 0 (dos cajas sin área) se devuelve 0 en vez de dividir por cero.
"""

import numpy as np


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    def area(r) -> float:
        return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])

    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def check() -> None:
    import torch
    from torchvision.ops import box_iou

    assert iou((0, 0, 2, 2), (0, 0, 2, 2)) == 1.0, "idénticas"
    assert iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0, "separadas"
    assert iou((0, 0, 1, 1), (1, 0, 2, 1)) == 0.0, "solo se tocan en un borde"
    assert iou((0, 0, 4, 4), (1, 1, 2, 2)) == 1 / 16, "una dentro de otra"
    assert iou((1, 1, 1, 1), (1, 1, 1, 1)) == 0.0, "sin área: 0, no NaN"
    rng = np.random.default_rng(0)
    for _ in range(200):
        xy = rng.random((2, 2)) * 10
        boxes = np.c_[xy, xy + rng.random((2, 2)) * 5]
        ref = box_iou(torch.tensor(boxes[:1]), torch.tensor(boxes[1:])).item()
        assert abs(iou(*boxes) - ref) < 1e-12
    print("ej02 ok — casos borde y 200 pares aleatorios contra torchvision")


if __name__ == "__main__":
    check()
