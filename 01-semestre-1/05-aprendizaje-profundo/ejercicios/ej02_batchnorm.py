"""Ejercicio 2: batch normalization. Ver README.md."""

import numpy as np
from applied_dl import Tensor


class BatchNorm2d:
    def __init__(self, channels: int, momentum: float = 0.1, eps: float = 1e-5):
        self.gamma = Tensor(np.ones((1, channels, 1, 1)), requires_grad=True)
        self.beta = Tensor(np.zeros((1, channels, 1, 1)), requires_grad=True)
        self.running_mean = np.zeros(channels)
        self.running_var = np.ones(channels)
        self.momentum, self.eps = momentum, eps
        self.training = True

    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError


def check() -> None:
    import torch

    rng = np.random.default_rng(0)
    x = rng.normal(2.0, 3.0, size=(8, 4, 5, 5))
    up = rng.normal(size=x.shape)
    bn = BatchNorm2d(4)
    xt = Tensor(x.copy(), requires_grad=True)
    out = bn(xt)  # una sola llamada: cada llamada en entrenamiento actualiza los promedios móviles
    (out * Tensor(up)).sum().backward()

    xr = torch.tensor(x, requires_grad=True)
    w, b = (
        torch.ones(4, dtype=torch.float64, requires_grad=True),
        torch.zeros(4, dtype=torch.float64, requires_grad=True),
    )
    rm, rv = torch.zeros(4, dtype=torch.float64), torch.ones(4, dtype=torch.float64)
    ref = torch.nn.functional.batch_norm(xr, rm, rv, w, b, training=True, momentum=0.1)
    (ref * torch.from_numpy(up)).sum().backward()

    np.testing.assert_allclose(out.data, ref.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(xt.grad, xr.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(bn.gamma.grad.ravel(), w.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(bn.running_mean, rm.numpy())
    np.testing.assert_allclose(bn.running_var, rv.numpy())
    bn.training = False
    ref_eval = torch.nn.functional.batch_norm(
        torch.from_numpy(x), rm, rv, w.detach(), b.detach(), training=False
    )
    np.testing.assert_allclose(bn(Tensor(x)).data, ref_eval.numpy(), atol=1e-10)
    print("ej02 ok — salida, gradientes y estadísticas móviles coinciden con torch")


if __name__ == "__main__":
    check()
