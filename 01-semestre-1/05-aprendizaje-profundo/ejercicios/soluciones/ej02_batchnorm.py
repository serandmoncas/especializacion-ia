"""Solución ejercicio 2: batch normalization.

Construida con operaciones del Tensor (media, resta, potencia, división), el backward sale gratis
del autograd, incluido el término sutil de que la media y la varianza dependen de *todo* el lote.
Por eso en entrenamiento el gradiente de un ejemplo depende de los demás del lote, y por eso
BN se comporta distinto (y peor) con lotes muy pequeños.
"""

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
        if self.training:
            n = x.shape[0] * x.shape[2] * x.shape[3]
            mean = x.sum(axis=(0, 2, 3), keepdims=True) * (1.0 / n)
            centered = x - mean
            var = (centered**2).sum(axis=(0, 2, 3), keepdims=True) * (
                1.0 / n
            )  # sesgada, como torch
            m = self.momentum
            self.running_mean = (1 - m) * self.running_mean + m * mean.data.ravel()
            self.running_var = (1 - m) * self.running_var + m * var.data.ravel() * n / (
                n - 1
            )  # insesgada
            x_hat = centered / (var + self.eps) ** 0.5
        else:
            mean = self.running_mean.reshape(1, -1, 1, 1)
            var = self.running_var.reshape(1, -1, 1, 1)
            x_hat = (x - mean) / np.sqrt(var + self.eps)
        return x_hat * self.gamma + self.beta


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
