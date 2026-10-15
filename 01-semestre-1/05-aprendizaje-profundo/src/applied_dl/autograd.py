"""Diferenciación automática en modo reverso sobre arreglos de NumPy.

Cada operación crea un `Tensor` que recuerda a sus padres y una función `_backward` que, dado
el gradiente de la salida, acumula el gradiente de cada padre (regla de la cadena local).
`backward()` ordena el grafo topológicamente y recorre esas funciones al revés: el costo es
proporcional al del forward, sin importar cuántos parámetros haya.

Detalle sutil: si una operación hizo *broadcasting* (p. ej. sumar un sesgo [d] a una matriz
[n, d]), el gradiente que llega tiene la forma expandida y hay que sumarlo sobre los ejes
expandidos para devolverlo a la forma original (`_unbroadcast`).
"""

from collections.abc import Callable

import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Suma `grad` sobre los ejes que el broadcasting agregó o expandió hasta llegar a `shape`."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    def __init__(
        self, data, requires_grad: bool = False, _parents: tuple["Tensor", ...] = (), _op: str = ""
    ):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None
        self._parents = _parents
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    # -- utilidades ---------------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __repr__(self) -> str:
        op = self._op or "hoja"
        return f"Tensor(shape={self.shape}, op={op}, requires_grad={self.requires_grad})"

    def _accumulate(self, g: np.ndarray) -> None:
        if self.requires_grad:
            g = _unbroadcast(g, self.shape)
            self.grad = g if self.grad is None else self.grad + g

    def _child(self, data: np.ndarray, parents: tuple["Tensor", ...], op: str) -> "Tensor":
        return Tensor(data, any(p.requires_grad for p in parents), parents, op)

    @staticmethod
    def _wrap(x) -> "Tensor":
        return x if isinstance(x, Tensor) else Tensor(x)

    def zero_grad(self) -> None:
        self.grad = None

    # -- backward -----------------------------------------------------------------------------
    def backward(self, grad: np.ndarray | None = None) -> None:
        """Propaga gradientes desde este tensor (por defecto, escalar con dL/dL = 1)."""
        order: list[Tensor] = []
        seen: set[int] = set()

        def visit(
            t: Tensor,
        ) -> None:  # DFS iterativo: grafos profundos (RNN) desbordan la recursión
            stack = [(t, False)]
            while stack:
                node, done = stack.pop()
                if done:
                    order.append(node)
                    continue
                if id(node) in seen:
                    continue
                seen.add(id(node))
                stack.append((node, True))
                stack.extend((p, False) for p in node._parents)

        visit(self)
        self.grad = np.ones_like(self.data) if grad is None else np.asarray(grad, dtype=np.float64)
        for node in reversed(order):
            # Nodos derivados solo de constantes (p. ej. -c en x - c) nunca reciben gradiente:
            # no hay nada que propagar desde ellos.
            if node.grad is not None:
                node._backward()

    # -- operaciones elementales -----------------------------------------------------------
    def __add__(self, other) -> "Tensor":
        other = self._wrap(other)
        out = self._child(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self._accumulate(out.grad)
            other._accumulate(out.grad)

        out._backward = _backward
        return out

    __radd__ = __add__

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other) -> "Tensor":
        return self + (-self._wrap(other))

    def __rsub__(self, other) -> "Tensor":
        return self._wrap(other) - self

    def __mul__(self, other) -> "Tensor":
        other = self._wrap(other)
        out = self._child(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self._accumulate(out.grad * other.data)
            other._accumulate(out.grad * self.data)

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other) -> "Tensor":
        return self * (self._wrap(other) ** -1.0)

    def __pow__(self, k: float) -> "Tensor":
        out = self._child(self.data**k, (self,), f"**{k}")

        def _backward() -> None:
            self._accumulate(out.grad * k * self.data ** (k - 1))

        out._backward = _backward
        return out

    def __matmul__(self, other) -> "Tensor":
        other = self._wrap(other)
        out = self._child(self.data @ other.data, (self, other), "@")

        def _backward() -> None:
            # C = A B  ⇒  dA = dC Bᵀ,  dB = Aᵀ dC
            self._accumulate(out.grad @ other.data.swapaxes(-1, -2))
            other._accumulate(self.data.swapaxes(-1, -2) @ out.grad)

        out._backward = _backward
        return out

    # -- funciones no lineales ------------------------------------------------------------------
    def relu(self) -> "Tensor":
        out = self._child(np.maximum(self.data, 0), (self,), "relu")

        def _backward() -> None:
            self._accumulate(out.grad * (self.data > 0))

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = self._child(t, (self,), "tanh")

        def _backward() -> None:
            self._accumulate(out.grad * (1 - t**2))

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        s = np.exp(-np.logaddexp(0, -self.data))
        out = self._child(s, (self,), "sigmoid")

        def _backward() -> None:
            self._accumulate(out.grad * s * (1 - s))

        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        e = np.exp(self.data)
        out = self._child(e, (self,), "exp")

        def _backward() -> None:
            self._accumulate(out.grad * e)

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = self._child(np.log(self.data), (self,), "log")

        def _backward() -> None:
            self._accumulate(out.grad / self.data)

        out._backward = _backward
        return out

    # -- reducciones y forma ----------------------------------------------------------------------
    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        out = self._child(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward() -> None:
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._accumulate(np.broadcast_to(g, self.shape).copy())

        out._backward = _backward
        return out

    def mean(self, axis: int | None = None, keepdims: bool = False) -> "Tensor":
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def reshape(self, *shape: int) -> "Tensor":
        out = self._child(self.data.reshape(*shape), (self,), "reshape")

        def _backward() -> None:
            self._accumulate(out.grad.reshape(self.shape))

        out._backward = _backward
        return out

    def __getitem__(self, idx) -> "Tensor":
        out = self._child(self.data[idx], (self,), "getitem")

        def _backward() -> None:
            g = np.zeros_like(self.data)
            np.add.at(g, idx, out.grad)  # add.at: índices repetidos acumulan
            self._accumulate(g)

        out._backward = _backward
        return out

    def log_softmax(self, axis: int = -1) -> "Tensor":
        """log softmax estable: x − max − log Σ exp(x − max)."""
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        logsm = shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        out = self._child(logsm, (self,), "log_softmax")

        def _backward() -> None:
            sm = np.exp(logsm)
            self._accumulate(out.grad - sm * out.grad.sum(axis=axis, keepdims=True))

        out._backward = _backward
        return out


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Entropía cruzada media para clasificación multiclase: −(1/n) Σ log softmax(z)[y].

    Su gradiente respecto a los logits es (softmax(z) − onehot(y))/n: limpio y estable, por eso
    se fusiona log_softmax con la selección de la clase correcta.
    """
    targets = np.asarray(targets)
    logp = logits.log_softmax(axis=-1)
    return -logp[np.arange(len(targets)), targets].mean()
