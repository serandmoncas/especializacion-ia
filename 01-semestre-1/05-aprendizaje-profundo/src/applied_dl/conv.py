"""Convolución 2D por im2col, integrada al autograd.

im2col reordena cada ventana de la entrada (C·kh·kw valores) como una columna: la convolución
se vuelve un único producto de matrices W[out, C·kh·kw] @ cols[C·kh·kw, N·H'·W'], que es como la
implementan las librerías sobre BLAS. El backward es otro par de productos más col2im, que
suma (con `np.add.at`) las contribuciones de ventanas solapadas.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from applied_dl.autograd import Tensor


def im2col(x: np.ndarray, kh: int, kw: int, stride: int) -> np.ndarray:
    """x [N, C, H, W] → [C·kh·kw, N·H'·W'] (sin copiar hasta el reshape final)."""
    win = sliding_window_view(x, (kh, kw), axis=(2, 3))[:, :, ::stride, ::stride]  # N,C,H',W',kh,kw
    n, c, ho, wo = win.shape[:4]
    return win.transpose(1, 4, 5, 0, 2, 3).reshape(c * kh * kw, n * ho * wo)


def col2im(cols: np.ndarray, x_shape: tuple[int, ...], kh: int, kw: int, stride: int) -> np.ndarray:
    n, c, h, w = x_shape
    ho, wo = (h - kh) // stride + 1, (w - kw) // stride + 1
    cols6 = cols.reshape(c, kh, kw, n, ho, wo)
    out = np.zeros(x_shape)
    for i in range(kh):
        for j in range(kw):
            out[:, :, i : i + stride * ho : stride, j : j + stride * wo : stride] += cols6[
                :, i, j
            ].transpose(1, 0, 2, 3)
    return out


def conv2d(
    x: Tensor, weight: Tensor, bias: Tensor | None = None, *, stride: int = 1, padding: int = 0
) -> Tensor:
    """Convolución (en realidad correlación cruzada, como en todas las librerías de DL).

    x [N, C, H, W], weight [O, C, kh, kw], bias [O] → [N, O, H', W'].
    """
    xd = (
        np.pad(x.data, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        if padding
        else x.data
    )
    o, _, kh, kw = weight.shape
    n = xd.shape[0]
    cols = im2col(xd, kh, kw, stride)
    ho, wo = (xd.shape[2] - kh) // stride + 1, (xd.shape[3] - kw) // stride + 1
    out_mat = weight.data.reshape(o, -1) @ cols  # [O, N·H'·W']
    if bias is not None:
        out_mat = out_mat + bias.data[:, None]
    data = out_mat.reshape(o, n, ho, wo).transpose(1, 0, 2, 3)
    parents = (x, weight) if bias is None else (x, weight, bias)
    out = Tensor(data, any(p.requires_grad for p in parents), parents, "conv2d")

    def _backward() -> None:
        g = out.grad.transpose(1, 0, 2, 3).reshape(o, -1)  # [O, N·H'·W']
        weight._accumulate((g @ cols.T).reshape(weight.shape))
        if bias is not None:
            bias._accumulate(g.sum(axis=1))
        if x.requires_grad:
            dx = col2im(weight.data.reshape(o, -1).T @ g, xd.shape, kh, kw, stride)
            if padding:
                dx = dx[:, :, padding:-padding, padding:-padding]
            x._accumulate(dx)

    out._backward = _backward
    return out
