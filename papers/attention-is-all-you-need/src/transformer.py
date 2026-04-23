"""Transformer encoder–decoder.

Paper §3.1–3.4. N=6 encoder + decoder layers (configurable). Each layer:
    x = LayerNorm(x + Sublayer(x))   (post-norm, original paper formulation)
Sublayers: multi-head attention, position-wise FFN.

Per §3.4 "we share the same weight matrix between the two embedding
layers and the pre-softmax linear transformation". Embeddings scaled by
√d_model.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .positional import PositionalEncoding


class PositionwiseFFN(nn.Module):
    """FFN(x) = max(0, x W_1 + b_1) W_2 + b_2  (paper §3.3)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                     # (B, L_tgt, d_model)
        memory: torch.Tensor,                # (B, L_src, d_model) — encoder output
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


def causal_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
    """Upper-triangular True mask: position i cannot attend to j > i."""
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)


class Transformer(nn.Module):
    """Encoder–decoder transformer with shared input/output embeddings (§3.4).

    Vocab is shared across source and target (paper uses ~37k shared BPE
    tokens for EN-DE). For EN-FR with separate vocabs, instantiate two
    embedding layers and don't share weights — but the paper's main config
    uses shared vocab.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.embed_dropout = nn.Dropout(dropout)

        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_layers)]
        )
        # Pre-softmax projection shares weights with embedding (§3.4).
        # We expose it as a property for forward(); no separate Parameter.

    def _pad_mask(self, x: torch.Tensor) -> torch.Tensor:
        # Returns (B, 1, 1, L) where True = PAD position (to be masked).
        return (x == self.pad_id).unsqueeze(1).unsqueeze(2)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.embed_dropout(self.pos_enc(x))
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.embed_dropout(self.pos_enc(x))
        for layer in self.decoder:
            x = layer(x, memory, tgt_mask, src_mask)
        return x

    def forward(
        self,
        src: torch.Tensor,                   # (B, L_src) token ids
        tgt: torch.Tensor,                   # (B, L_tgt) token ids
    ) -> torch.Tensor:
        """Returns logits over vocabulary, shape (B, L_tgt, vocab_size)."""
        src_pad = self._pad_mask(src)                                # (B, 1, 1, L_src)
        tgt_pad = self._pad_mask(tgt)                                # (B, 1, 1, L_tgt)
        causal = causal_mask(tgt.size(1), device=tgt.device)         # (L_tgt, L_tgt)
        tgt_mask = tgt_pad | causal.unsqueeze(0).unsqueeze(0)        # broadcast → (B, 1, L_tgt, L_tgt)

        memory = self.encode(src, src_pad)
        out = self.decode(tgt, memory, tgt_mask, src_pad)            # (B, L_tgt, d_model)
        # Pre-softmax linear shares weights with input embedding (§3.4).
        return out @ self.embedding.weight.T                         # (B, L_tgt, vocab_size)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
