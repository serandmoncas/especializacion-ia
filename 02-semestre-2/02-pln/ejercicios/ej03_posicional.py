"""Ejercicio 3: codificación posicional sinusoidal (Vaswani et al., 2017). Ver README.md."""

import numpy as np


def sinusoidal_positions(n: int, d: int) -> np.ndarray:
    """PE[pos, 2i] = sin(pos / 10000^(2i/d)), PE[pos, 2i+1] = cos(…); d par."""
    raise NotImplementedError


def check() -> None:
    pe = sinusoidal_positions(50, 4)
    np.testing.assert_allclose(pe[0], [0, 1, 0, 1])
    np.testing.assert_allclose(pe[1], [np.sin(1), np.cos(1), np.sin(0.01), np.cos(0.01)])
    big = sinusoidal_positions(200, 64)
    # El producto punto solo depende de la distancia: <PE[p], PE[p+k]> no depende de p.
    for k in (1, 5, 20):
        dots = [big[p] @ big[p + k] for p in range(0, 150, 10)]
        np.testing.assert_allclose(dots, dots[0], atol=1e-9)
    print("ej03 ok — valores a mano y producto punto invariante a traslaciones")


if __name__ == "__main__":
    check()
