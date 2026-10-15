"""Atención escalada y multi-cabeza en NumPy (la referencia en los tests es `torch`)."""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax estable: resta el máximo antes de exponenciar."""
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """softmax(q·kᵀ/√d)·v. `mask` booleano: True = se puede atender. Devuelve (salida, pesos)."""
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores, axis=-1)
    return weights @ v, weights


def causal_mask(n: int) -> np.ndarray:
    """Cada posición atiende solo a sí misma y a las anteriores."""
    return np.tril(np.ones((n, n), dtype=bool))


def multi_head_attention(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    num_heads: int,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Auto-atención con `num_heads` cabezas; `x` de forma (..., n, d) y proyecciones `x @ w`."""
    *lead, n, d = x.shape
    if d % num_heads:
        raise ValueError(f"d = {d} no es divisible entre {num_heads} cabezas")
    h = d // num_heads

    def split(t: np.ndarray) -> np.ndarray:  # (..., n, d) -> (..., cabezas, n, h)
        return np.swapaxes(t.reshape(*lead, n, num_heads, h), -2, -3)

    out, weights = scaled_dot_product_attention(
        split(x @ w_q), split(x @ w_k), split(x @ w_v), mask
    )
    out = np.swapaxes(out, -2, -3).reshape(*lead, n, d)
    return out @ w_o, weights
