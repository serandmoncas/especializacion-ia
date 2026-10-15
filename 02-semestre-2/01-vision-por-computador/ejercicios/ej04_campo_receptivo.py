"""Ejercicio 4: tamaño de salida y campo receptivo de una pila de convoluciones. Ver README.md."""

import numpy as np


def conv_output_size(n: int, k: int, s: int = 1, p: int = 0, d: int = 1) -> int:
    """Tamaño de salida de una convolución."""
    raise NotImplementedError


def receptive_field(layers: list[tuple[int, int, int]]) -> int:
    """Campo receptivo de una pila de capas [(kernel, stride, dilation), ...]."""
    raise NotImplementedError


def check() -> None:
    import torch

    for n, k, s, p, d in [(32, 3, 1, 1, 1), (32, 3, 2, 1, 1), (31, 5, 2, 0, 1), (64, 3, 1, 2, 2)]:
        conv = torch.nn.Conv2d(1, 1, k, stride=s, padding=p, dilation=d)
        assert conv(torch.zeros(1, 1, n, n)).shape[-1] == conv_output_size(n, k, s, p, d)
    for layers in (
        [(3, 1, 1), (3, 1, 1)],
        [(3, 2, 1), (3, 1, 1)],
        [(7, 2, 1), (3, 2, 1), (3, 1, 1), (3, 1, 1)],
        [(3, 1, 2), (3, 1, 4)],
    ):
        convs = [torch.nn.Conv2d(1, 1, k, stride=s, dilation=d, bias=False) for k, s, d in layers]
        for c in convs:
            torch.nn.init.constant_(c.weight, 1.0)
        x = torch.ones(1, 1, 64, 64, requires_grad=True)
        y = torch.nn.Sequential(*convs)(x)
        y[0, 0, y.shape[2] // 2, y.shape[3] // 2].backward()
        rows = np.flatnonzero(x.grad[0, 0].sum(dim=1).numpy())
        assert rows.max() - rows.min() + 1 == receptive_field(layers), layers
    print("ej04 ok — fórmulas = tamaños de torch y campo receptivo medido con gradientes")


if __name__ == "__main__":
    check()
