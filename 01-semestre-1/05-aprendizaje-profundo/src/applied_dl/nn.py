"""Redes neuronales sobre el autograd propio: capas, inicialización, dropout y optimizadores."""

from collections.abc import Iterable
from itertools import pairwise

import numpy as np

from applied_dl.autograd import Tensor


class Module:
    training = True

    def parameters(self) -> list[Tensor]:
        params = []
        for v in vars(self).values():
            if isinstance(v, Tensor) and v.requires_grad:
                params.append(v)
            elif isinstance(v, Module):
                params += v.parameters()
            elif isinstance(v, list):
                params += [p for m in v if isinstance(m, Module) for p in m.parameters()]
        return params

    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        for v in vars(self).values():
            for m in v if isinstance(v, list) else [v]:
                if isinstance(m, Module):
                    m.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()


def init_weights(fan_in: int, fan_out: int, scheme: str, rng: np.random.Generator) -> np.ndarray:
    """Inicialización que preserva la varianza de las activaciones capa a capa.

    - "xavier" (Glorot): Var(W) = 2/(fan_in + fan_out). Pensada para tanh/sigmoide.
    - "he" (Kaiming):    Var(W) = 2/fan_in. ReLU anula la mitad de las entradas; el 2 lo compensa.
    - "naive":           N(0, 1). Las activaciones crecen o se anulan exponencialmente
                         con la profundidad.
    """
    if scheme == "xavier":
        std = np.sqrt(2.0 / (fan_in + fan_out))
    elif scheme == "he":
        std = np.sqrt(2.0 / fan_in)
    elif scheme == "naive":
        std = 1.0
    else:
        raise ValueError(f"esquema desconocido: {scheme!r}")
    return rng.normal(0.0, std, size=(fan_in, fan_out))


class Linear(Module):
    def __init__(
        self, fan_in: int, fan_out: int, *, init: str = "he", rng: np.random.Generator | None = None
    ):
        rng = rng or np.random.default_rng(0)
        self.W = Tensor(init_weights(fan_in, fan_out, init, rng), requires_grad=True)
        self.b = Tensor(np.zeros(fan_out), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b


class Dropout(Module):
    """Dropout invertido: en entrenamiento anula cada unidad con prob. p y escala por 1/(1−p),
    para que en evaluación (sin dropout) la activación esperada sea la misma."""

    def __init__(self, p: float = 0.5, rng: np.random.Generator | None = None):
        self.p = p
        self.rng = rng or np.random.default_rng(0)

    def __call__(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        mask = (self.rng.random(x.shape) >= self.p) / (1 - self.p)
        return x * mask


ACTIVATIONS = {"relu": Tensor.relu, "tanh": Tensor.tanh, "sigmoid": Tensor.sigmoid}


class MLP(Module):
    """Perceptrón multicapa: [Linear → activación → Dropout] × (L−1) → Linear (logits)."""

    def __init__(
        self,
        sizes: list[int],
        *,
        activation: str = "relu",
        init: str = "he",
        dropout: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        rng = rng or np.random.default_rng(0)
        self.layers = [Linear(a, b, init=init, rng=rng) for a, b in pairwise(sizes)]
        self.drops = [Dropout(dropout, rng) for _ in self.layers[:-1]]
        self.activation = activation

    def __call__(self, x: Tensor, *, return_hidden: bool = False):
        act = ACTIVATIONS[self.activation]
        hidden = []
        for layer, drop in zip(self.layers[:-1], self.drops, strict=True):
            x = drop(act(layer(x)))
            hidden.append(x)
        out = self.layers[-1](x)
        return (out, hidden) if return_hidden else out


class SGD:
    """v ← μv + g;  θ ← θ − η v  (convención de PyTorch). weight_decay suma λθ al gradiente."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        self.params = list(params)
        self.lr, self.momentum, self.weight_decay = lr, momentum, weight_decay
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for p, v in zip(self.params, self.velocity, strict=True):
            if p.grad is None:
                continue
            g = p.grad + self.weight_decay * p.data
            v *= self.momentum
            v += g
            p.data -= self.lr * v

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()


class Adam:
    """Adam con corrección de sesgo; `decoupled=True` es AdamW (Loshchilov & Hutter, 2019).

    m ← β₁m + (1−β₁)g,  v ← β₂v + (1−β₂)g²,  θ ← θ − η m̂/(√v̂ + ε).
    Con L2 clásico (decoupled=False), λθ entra en g y queda dividido por √v̂: los pesos con
    gradientes grandes se regularizan menos. AdamW resta ηλθ directamente, igual para todos.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decoupled: bool = False,
    ):
        self.params = list(params)
        self.lr, self.betas, self.eps = lr, betas, eps
        self.weight_decay, self.decoupled = weight_decay, decoupled
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.betas
        for p, m, v in zip(self.params, self.m, self.v, strict=True):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay and not self.decoupled:
                g = g + self.weight_decay * p.data
            if self.weight_decay and self.decoupled:
                p.data -= self.lr * self.weight_decay * p.data
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g**2
            m_hat = m / (1 - b1**self.t)
            v_hat = v / (1 - b2**self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()
