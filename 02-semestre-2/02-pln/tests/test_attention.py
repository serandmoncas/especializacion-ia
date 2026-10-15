"""Atención propia contra `torch.nn.functional.scaled_dot_product_attention` y `MultiheadAttention`."""

import numpy as np
import pytest
from research_paths.attention import (
    causal_mask,
    multi_head_attention,
    scaled_dot_product_attention,
    softmax,
)


def test_softmax_rows_sum_to_one_and_are_stable():
    p = softmax(np.array([[1000.0, 1000.0], [0.0, -np.inf]]))
    np.testing.assert_allclose(p, [[0.5, 0.5], [1.0, 0.0]])


def test_causal_mask_is_lower_triangular():
    np.testing.assert_array_equal(causal_mask(3), [[1, 0, 0], [1, 1, 0], [1, 1, 1]])


def test_causal_attention_ignores_the_future():
    rng = np.random.default_rng(0)
    q, k, v = (rng.normal(size=(4, 3)) for _ in range(3))
    _, w = scaled_dot_product_attention(q, k, v, causal_mask(4))
    assert np.all(w[np.triu_indices(4, 1)] == 0)
    np.testing.assert_allclose(w.sum(-1), 1)


def test_matches_torch_sdpa():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    q, k, v = (rng.normal(size=(2, 5, 8)) for _ in range(3))
    for mask, causal in ((None, False), (causal_mask(5), True)):
        out, _ = scaled_dot_product_attention(q, k, v, mask)
        ref = torch.nn.functional.scaled_dot_product_attention(
            *(torch.tensor(t) for t in (q, k, v)), is_causal=causal
        )
        np.testing.assert_allclose(out, ref.numpy(), atol=1e-10)


def test_multi_head_matches_torch_multihead_attention():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    d, heads = 8, 2
    x = rng.normal(size=(3, 5, d))
    w_q, w_k, w_v, w_o = (rng.normal(size=(d, d)) for _ in range(4))
    mha = torch.nn.MultiheadAttention(d, heads, bias=False, batch_first=True, dtype=torch.float64)
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.tensor(np.concatenate([w_q.T, w_k.T, w_v.T])))
        mha.out_proj.weight.copy_(torch.tensor(w_o.T))
    xt = torch.tensor(x)
    ref, ref_w = mha(xt, xt, xt, average_attn_weights=False)
    out, w = multi_head_attention(x, w_q, w_k, w_v, w_o, heads)
    np.testing.assert_allclose(out, ref.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(w, ref_w.detach().numpy(), atol=1e-10)


def test_multi_head_rejects_indivisible_heads():
    x = np.zeros((2, 6))
    w = np.eye(6)
    with pytest.raises(ValueError, match="divisible"):
        multi_head_attention(x, w, w, w, w, num_heads=4)
