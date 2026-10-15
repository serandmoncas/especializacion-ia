"""Ejercicio 1: max pooling con backward. Ver README.md."""

import numpy as np
from applied_dl import Tensor


def max_pool2d(x: Tensor, k: int) -> Tensor:
    raise NotImplementedError


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
