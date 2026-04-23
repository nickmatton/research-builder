"""Transformer model (Vaswani et al., 2017 - 'Attention Is All You Need')."""

import math
import torch
import torch.nn as nn

from config import TransformerConfig
from attention import MultiHeadAttention
from feed_forward import PositionwiseFeedForward
from positional_encoding import PositionalEncoding


class SublayerConnection(nn.Module):
    """Residual connection followed by layer normalization.

    Implements: LayerNorm(x + Sublayer(x))
    This is post-norm as described in the paper.
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        """Apply residual connection and layer norm.

        Args:
            x: Input tensor (residual)
            sublayer_output: Output of the sublayer function

        Returns:
            LayerNorm(x + Dropout(sublayer_output))
        """
        return self.norm(x + self.dropout(sublayer_output))


class EncoderLayer(nn.Module):
    """Single encoder layer.

    Contains:
    1. Multi-head self-attention
    2. Position-wise feed-forward network
    Both with residual connections and layer normalization.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            config.d_model, config.n_heads, config.d_k, config.d_v, config.dropout
        )
        self.feed_forward = PositionwiseFeedForward(
            config.d_model, config.d_ff, config.dropout
        )
        self.sublayer1 = SublayerConnection(config.d_model, config.dropout)
        self.sublayer2 = SublayerConnection(config.d_model, config.dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, d_model)
            src_mask: Source mask for padding, shape broadcastable to attention scores.

        Returns:
            (batch, seq_len, d_model)
        """
        attn_out = self.self_attn(x, x, x, mask=src_mask)
        x = self.sublayer1(x, attn_out)
        ff_out = self.feed_forward(x)
        x = self.sublayer2(x, ff_out)
        return x


class DecoderLayer(nn.Module):
    """Single decoder layer.

    Contains:
    1. Masked multi-head self-attention (causal)
    2. Multi-head encoder-decoder attention
    3. Position-wise feed-forward network
    All with residual connections and layer normalization.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            config.d_model, config.n_heads, config.d_k, config.d_v, config.dropout
        )
        self.cross_attn = MultiHeadAttention(
            config.d_model, config.n_heads, config.d_k, config.d_v, config.dropout
        )
        self.feed_forward = PositionwiseFeedForward(
            config.d_model, config.d_ff, config.dropout
        )
        self.sublayer1 = SublayerConnection(config.d_model, config.dropout)
        self.sublayer2 = SublayerConnection(config.d_model, config.dropout)
        self.sublayer3 = SublayerConnection(config.d_model, config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, tgt_seq_len, d_model) - decoder input
            encoder_output: (batch, src_seq_len, d_model)
            src_mask: Source padding mask
            tgt_mask: Target causal + padding mask

        Returns:
            (batch, tgt_seq_len, d_model)
        """
        # 1. Masked self-attention
        self_attn_out = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.sublayer1(x, self_attn_out)

        # 2. Encoder-decoder attention (queries from decoder, keys/values from encoder)
        cross_attn_out = self.cross_attn(x, encoder_output, encoder_output, mask=src_mask)
        x = self.sublayer2(x, cross_attn_out)

        # 3. Feed-forward
        ff_out = self.feed_forward(x)
        x = self.sublayer3(x, ff_out)
        return x


class Encoder(nn.Module):
    """Stack of N encoder layers."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N decoder layers."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """Full Transformer model for sequence-to-sequence tasks.

    Implements the encoder-decoder architecture from 'Attention Is All You Need'.

    Key features:
    - Shared embedding weights between source, target, and output projection
    - Sinusoidal positional encoding
    - Causal masking in decoder
    - Post-norm residual connections: LayerNorm(x + Sublayer(x))
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        # Shared embedding (Section 3.4)
        # Same weight matrix for src embedding, tgt embedding, and pre-softmax linear
        self.embedding = nn.Embedding(config.src_vocab_size, config.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.d_model, config.dropout, config.max_seq_len)

        # Encoder and decoder stacks
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        # Output projection shares weights with embedding
        self.output_projection = nn.Linear(config.d_model, config.tgt_vocab_size, bias=False)
        self.output_projection.weight = self.embedding.weight  # Weight tying

        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters using Xavier uniform (common for Transformers)."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def generate_causal_mask(sz: int, device: torch.device | None = None) -> torch.Tensor:
        """Generate causal (subsequent) mask for decoder self-attention.

        Returns a (sz, sz) mask where mask[i, j] = 1 if j <= i, else 0.
        This prevents attending to future positions.
        """
        mask = torch.tril(torch.ones(sz, sz, device=device)).bool()
        return mask  # (sz, sz) - broadcastable to (batch, heads, sz, sz)

    @staticmethod
    def generate_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """Generate padding mask.

        Args:
            seq: (batch, seq_len) token indices
            pad_idx: Index of the padding token

        Returns:
            (batch, 1, 1, seq_len) mask where 1 = valid, 0 = padding
        """
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)

    def encode(
        self, src: torch.Tensor, src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode source sequence.

        Args:
            src: (batch, src_len) source token indices
            src_mask: Source padding mask

        Returns:
            (batch, src_len, d_model) encoder output
        """
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        return self.encoder(x, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode target sequence.

        Args:
            tgt: (batch, tgt_len) target token indices
            encoder_output: (batch, src_len, d_model)
            src_mask: Source padding mask
            tgt_mask: Target causal + padding mask

        Returns:
            (batch, tgt_len, d_model) decoder output
        """
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        return self.decoder(x, encoder_output, src_mask, tgt_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full forward pass.

        Args:
            src: (batch, src_len) source token indices
            tgt: (batch, tgt_len) target token indices
            src_mask: Source padding mask, (batch, 1, 1, src_len)
            tgt_mask: Target mask combining causal + padding,
                      (batch, 1, tgt_len, tgt_len)

        Returns:
            (batch, tgt_len, vocab_size) logits
        """
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        logits = self.output_projection(decoder_output)
        return logits
