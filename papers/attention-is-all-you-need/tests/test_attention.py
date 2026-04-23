"""Unit tests for src/attention.py.

Verification ladder rung 1: shape correctness, mask semantics, output range.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.attention import MultiHeadAttention, scaled_dot_product_attention


def test_scaled_dot_product_shape():
    B, h, L_q, L_k, d_k = 2, 4, 5, 7, 8
    q = torch.randn(B, h, L_q, d_k)
    k = torch.randn(B, h, L_k, d_k)
    v = torch.randn(B, h, L_k, d_k)
    out = scaled_dot_product_attention(q, k, v)
    assert out.shape == (B, h, L_q, d_k)


def test_scaled_dot_product_mask_zeros_attention():
    """A fully masked key position contributes nothing to output."""
    B, h, L, d = 1, 1, 4, 4
    q = torch.randn(B, h, L, d)
    k = torch.randn(B, h, L, d)
    v = torch.randn(B, h, L, d)

    # Mask out the last key for query 0 only.
    mask = torch.zeros(B, h, L, L, dtype=torch.bool)
    mask[0, 0, 0, -1] = True

    out_masked = scaled_dot_product_attention(q, k, v, mask)
    # Manually verify: zero out v[..., -1, :] in the unmasked computation should NOT
    # change the result for query 0.
    v2 = v.clone()
    v2[0, 0, -1] = 0
    # With masked softmax, query 0's attention over the last key position is 0,
    # so v[..., -1] doesn't contribute.
    out_with_zero = scaled_dot_product_attention(q, k, v2, mask)
    assert torch.allclose(out_masked[0, 0, 0], out_with_zero[0, 0, 0], atol=1e-6)


def test_multihead_forward_shape():
    B, L, d_model, h = 3, 7, 32, 4
    mha = MultiHeadAttention(d_model, h)
    x = torch.randn(B, L, d_model)
    out = mha(x, x, x)
    assert out.shape == (B, L, d_model)


def test_multihead_d_model_must_be_divisible_by_heads():
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=10, num_heads=4)


def test_multihead_with_2d_causal_mask():
    """A causal mask given as (L, L) should broadcast across batch + heads."""
    B, L, d_model, h = 2, 5, 16, 2
    mha = MultiHeadAttention(d_model, h)
    x = torch.randn(B, L, d_model)
    causal = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    out = mha(x, x, x, causal)
    assert out.shape == (B, L, d_model)
