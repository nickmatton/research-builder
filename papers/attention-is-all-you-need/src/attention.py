"""Scaled dot-product + multi-head attention.

Paper §3.2 (p.4–5):
    Attention(Q, K, V) = softmax(Q K^T / √d_k) V

Multi-head splits Q/K/V into h heads of d_k = d_v = d_model / h, runs
attention per head, concatenates, and projects back through W^O.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,            # (B, h, L_q, d_k)
    k: torch.Tensor,            # (B, h, L_k, d_k)
    v: torch.Tensor,            # (B, h, L_k, d_v)
    mask: torch.Tensor | None = None,  # (B, 1, L_q, L_k) or broadcastable; True = MASKED
) -> torch.Tensor:
    """Returns (B, h, L_q, d_v)."""
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with shared d_k = d_v = d_model / h."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, d_model) → (B, h, L, d_k)
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, h, L, d_k) → (B, L, d_model)
        B, _, L, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, self.d_model)

    def forward(
        self,
        query: torch.Tensor,    # (B, L_q, d_model)
        key: torch.Tensor,      # (B, L_k, d_model)
        value: torch.Tensor,    # (B, L_k, d_model)
        mask: torch.Tensor | None = None,  # broadcastable to (B, h, L_q, L_k); True = MASKED
    ) -> torch.Tensor:
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        # If mask is (B, L_q, L_k), add the heads dim.
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)
        # If mask is (L_q, L_k) (e.g. causal), broadcast across batch + heads.
        elif mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        out = scaled_dot_product_attention(q, k, v, mask)  # (B, h, L_q, d_v)
        return self.w_o(self._merge_heads(out))
