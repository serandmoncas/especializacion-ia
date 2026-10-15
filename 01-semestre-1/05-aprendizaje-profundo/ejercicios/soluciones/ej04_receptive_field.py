"""Solución ejercicio 4: campo receptivo.

Cada capa con kernel k agrega (k − 1)·j pixeles de entrada al campo receptivo, donde j (el “salto”)
es la distancia en pixeles de entrada entre unidades vecinas de esa capa; un pooling de 2
duplica j. Por eso las capas profundas ven regiones grandes con kernels pequeños.
"""

import numpy as np


def receptive_field(layers: list[tuple[int, int]]) -> list[int]:
    """layers: (kernel, stride) en orden. Devuelve el campo receptivo tras cada capa."""
    r, j, out = 1, 1, []
    for k, s in layers:
        r += (k - 1) * j
        j *= s
        out.append(r)
    return out


# CNN del proyecto: [conv3, conv3, pool2] × 2
PROJECT_CNN = [(3, 1), (3, 1), (2, 2), (3, 1), (3, 1), (2, 2)]


def empirical_receptive_field(seed: int = 0) -> int:
    """Ancho de la región de entrada con gradiente ≠ 0 para una unidad central de la última conv."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    convs = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 4, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(4, 4, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 4, 3, padding=1),
        nn.ReLU(),
    ).double()
    x = torch.randn(1, 1, 56, 56, dtype=torch.float64, requires_grad=True)
    out = convs(x)  # [1, 4, 28, 28]
    out[0, :, 14, 14].sum().backward()
    cols = np.flatnonzero(np.abs(x.grad[0, 0].numpy()).sum(axis=0) > 0)
    return int(cols.max() - cols.min() + 1)


def check() -> None:
    rf = receptive_field(PROJECT_CNN)
    assert rf == [3, 5, 6, 10, 14, 16], rf
    # hasta la última conv (sin el pool final): 14
    emp = empirical_receptive_field()
    assert emp == rf[4], (emp, rf[4])
    print(f"ej04 ok — campo receptivo por capa {rf}; verificado con gradientes: {emp} pixeles")


if __name__ == "__main__":
    check()
