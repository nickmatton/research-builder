"""Model configuration for the Transformer (Vaswani et al., 2017)."""

from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Configuration for the Transformer model.

    Provides both 'base' and 'big' configurations from Table 3 of the paper.
    """
    # Vocabulary
    src_vocab_size: int = 37000
    tgt_vocab_size: int = 37000

    # Architecture
    n_layers: int = 6          # N: number of encoder/decoder layers
    d_model: int = 512         # model dimension
    d_ff: int = 2048           # feed-forward inner dimension
    n_heads: int = 8           # h: number of attention heads
    d_k: int = 64              # key dimension per head
    d_v: int = 64              # value dimension per head

    # Regularization
    dropout: float = 0.1       # P_drop

    # Sequence length
    max_seq_len: int = 5000    # maximum sequence length for positional encoding

    @classmethod
    def base(cls, src_vocab_size: int = 37000, tgt_vocab_size: int = 37000) -> "TransformerConfig":
        """Base model configuration (~65M parameters)."""
        return cls(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            n_layers=6,
            d_model=512,
            d_ff=2048,
            n_heads=8,
            d_k=64,
            d_v=64,
            dropout=0.1,
        )

    @classmethod
    def big(cls, src_vocab_size: int = 37000, tgt_vocab_size: int = 37000) -> "TransformerConfig":
        """Big model configuration (~213M parameters)."""
        return cls(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            n_layers=6,
            d_model=1024,
            d_ff=4096,
            n_heads=16,
            d_k=64,
            d_v=64,
            dropout=0.3,
        )
