"""Solución ejercicio 3: atención con producto escalar escalado.

Si q y k tienen componentes independientes de varianza 1, q·k tiene varianza d. Sin dividir por
√d, para d grande los logits del softmax son enormes, el softmax se vuelve casi one-hot (entropía
≈ 0) y su gradiente casi nulo: la atención deja de aprender. La escala devuelve la varianza a 1.
"""

import numpy as np
from applied_dl import Tensor


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return x.log_softmax(axis=axis).exp()


def attention(q: Tensor, k: Tensor, v: Tensor, *, causal: bool = False) -> Tensor:
    """q, k, v: [..., T, d] → [..., T, d]."""
    d = q.shape[-1]
    scores = (q @ _transpose(k)) * (1 / np.sqrt(d))
    if causal:
        t = q.shape[-2]
        mask = np.triu(np.full((t, t), -np.inf), k=1)  # posición i no ve j > i
        scores = scores + Tensor(mask)
    return softmax(scores) @ v


def _transpose(x: Tensor) -> Tensor:
    out = Tensor(x.data.swapaxes(-1, -2), x.requires_grad, (x,), "T")

    def _backward() -> None:
        x._accumulate(out.grad.swapaxes(-1, -2))

    out._backward = _backward
    return out


def softmax_entropy(d: int, scaled: bool, n: int = 2000, t: int = 16, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    q, k = rng.normal(size=(n, 1, d)), rng.normal(size=(n, t, d))
    s = (q @ k.swapaxes(-1, -2))[:, 0] / (np.sqrt(d) if scaled else 1)
    p = np.exp(s - s.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return float(-(p * np.log(p + 1e-300)).sum(1).mean())


def check() -> None:
    import torch

    rng = np.random.default_rng(0)
    q, k, v = (rng.normal(size=(2, 6, 8)) for _ in range(3))
    up = rng.normal(size=(2, 6, 8))
    for causal in (False, True):
        qt, kt, vt = (Tensor(a.copy(), requires_grad=True) for a in (q, k, v))
        (attention(qt, kt, vt, causal=causal) * Tensor(up)).sum().backward()
        qr, kr, vr = (torch.tensor(a, requires_grad=True) for a in (q, k, v))
        ref = torch.nn.functional.scaled_dot_product_attention(qr, kr, vr, is_causal=causal)
        (ref * torch.from_numpy(up)).sum().backward()
        np.testing.assert_allclose(
            attention(Tensor(q), Tensor(k), Tensor(v), causal=causal).data,
            ref.detach().numpy(),
            atol=1e-10,
        )
        for ours, theirs in ((qt, qr), (kt, kr), (vt, vr)):
            np.testing.assert_allclose(ours.grad, theirs.grad.numpy(), atol=1e-10)
    max_entropy = np.log(16)
    table = {d: (softmax_entropy(d, False), softmax_entropy(d, True)) for d in (8, 64, 512)}
    assert table[512][0] < 0.2 * max_entropy < table[512][1]
    print(
        "ej03 ok — coincide con torch. Entropía del softmax (máx. "
        f"{max_entropy:.2f}) sin/con escala:",
        {d: (round(a, 2), round(b, 2)) for d, (a, b) in table.items()},
    )


if __name__ == "__main__":
    check()
