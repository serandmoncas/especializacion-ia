"""Ejercicio 2: máscara de relleno (padding) en la atención. Ver README.md."""

import numpy as np


def padding_mask(lengths: list[int], n: int) -> np.ndarray:
    """(lote, 1, n) booleana: True en las posiciones reales de cada secuencia."""
    return (np.arange(n)[None, :] < np.asarray(lengths)[:, None])[:, None, :]


def masked_attention(q, k, v, lengths):
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    scores = np.where(padding_mask(lengths, k.shape[-2]), scores, -np.inf)
    scores = scores - scores.max(-1, keepdims=True)
    w = np.exp(scores)
    return (w / w.sum(-1, keepdims=True)) @ v


def check() -> None:
    import torch

    rng = np.random.default_rng(0)
    q, k, v = (rng.normal(size=(3, 5, 4)) for _ in range(3))
    lengths = [5, 3, 1]
    mask = torch.tensor(padding_mask(lengths, 5))
    ref = torch.nn.functional.scaled_dot_product_attention(
        *(torch.tensor(t) for t in (q, k, v)), attn_mask=mask
    )
    np.testing.assert_allclose(masked_attention(q, k, v, lengths), ref.numpy(), atol=1e-10)
    out = masked_attention(q, k, v, [1, 1, 1])
    np.testing.assert_allclose(out, np.broadcast_to(v[:, :1, :], out.shape), atol=1e-12)
    print("ej02 ok — igual que scaled_dot_product_attention con attn_mask")


if __name__ == "__main__":
    check()
