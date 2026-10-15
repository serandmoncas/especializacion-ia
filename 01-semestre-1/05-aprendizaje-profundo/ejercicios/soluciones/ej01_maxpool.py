"""Solución ejercicio 1: max pooling con backward.

Reorganizar x [N, C, H, W] en ventanas [N, C, H/k, W/k, k·k] permite tomar el máximo y su argmax
con una sola operación. El backward es un enrutamiento: cada gradiente de salida se escribe en la
posición del argmax de su ventana y el resto queda en cero. Por eso el pooling no tiene parámetros
pero sí “elige” qué pixeles reciben señal de aprendizaje.
"""

import numpy as np
from applied_dl import Tensor


def max_pool2d(x: Tensor, k: int) -> Tensor:
    n, c, h, w = x.shape
    if h % k or w % k:
        raise ValueError("H y W deben ser múltiplos de k")
    win = (
        x.data.reshape(n, c, h // k, k, w // k, k)
        .transpose(0, 1, 2, 4, 3, 5)
        .reshape(n, c, h // k, w // k, k * k)
    )
    arg = win.argmax(axis=-1)
    out = Tensor(
        np.take_along_axis(win, arg[..., None], axis=-1)[..., 0], x.requires_grad, (x,), "maxpool"
    )

    def _backward() -> None:
        g = np.zeros_like(win)
        np.put_along_axis(g, arg[..., None], out.grad[..., None], axis=-1)
        g = g.reshape(n, c, h // k, w // k, k, k).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h, w)
        x._accumulate(g)

    out._backward = _backward
    return out


def check() -> None:
    import torch

    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 8, 6))
    up = rng.normal(size=(2, 3, 4, 3))
    xt = Tensor(x.copy(), requires_grad=True)
    out = max_pool2d(xt, 2)
    (out * Tensor(up)).sum().backward()
    xr = torch.tensor(x, requires_grad=True)
    ref = torch.nn.functional.max_pool2d(xr, 2)
    (ref * torch.from_numpy(up)).sum().backward()
    np.testing.assert_allclose(out.data, ref.detach().numpy())
    np.testing.assert_allclose(xt.grad, xr.grad.numpy())
    assert np.count_nonzero(xt.grad) == up.size, "un solo pixel por ventana recibe gradiente"
    print("ej01 ok — forward y backward coinciden con torch")


if __name__ == "__main__":
    check()
