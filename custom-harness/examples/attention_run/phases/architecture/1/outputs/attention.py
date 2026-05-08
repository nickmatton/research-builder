"""Multi-head attention mechanism (Section 3.2)."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled Dot-Product Attention (Equation 1).

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Args:
        query: (batch, heads, seq_len_q, d_k)
        key:   (batch, heads, seq_len_k, d_k)
        value: (batch, heads, seq_len_k, d_v)
        mask:  Broadcastable mask. Positions with True/1 are *allowed*;
               positions with False/0 are masked to -inf.
        dropout: Optional dropout on attention weights.

    Returns:
        Tuple of (output, attention_weights).
        output: (batch, heads, seq_len_q, d_v)
        attention_weights: (batch, heads, seq_len_q, seq_len_k)
    """
    d_k = query.size(-1)
    # (batch, heads, seq_q, seq_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    output = torch.matmul(attn_weights, value)
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention (Section 3.2.2).

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)

    Uses linear projections without bias (common practice; paper does not
    explicitly mention bias terms for attention projections).
    """

    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        # Projection matrices (combined for all heads)
        self.w_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.w_o = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)
        self.attn_weights = None  # Store for visualization

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query: (batch, seq_len_q, d_model)
            key:   (batch, seq_len_k, d_model)
            value: (batch, seq_len_k, d_model)
            mask:  Optional mask, broadcastable to (batch, 1, seq_len_q, seq_len_k)
                   or (batch, heads, seq_len_q, seq_len_k).
                   1 = attend, 0 = mask out.

        Returns:
            (batch, seq_len_q, d_model)
        """
        batch_size = query.size(0)

        # Linear projections and reshape to (batch, heads, seq_len, d_k/d_v)
        q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # Apply attention
        x, self.attn_weights = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)

        # Concatenate heads and apply output projection
        # (batch, heads, seq_len, d_v) -> (batch, seq_len, heads * d_v)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)

        return self.w_o(x)
