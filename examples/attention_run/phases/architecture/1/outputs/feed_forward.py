"""Position-wise Feed-Forward Network (Section 3.3)."""

import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network.

    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

    Two linear transformations with a ReLU activation in between.
    Applied to each position separately and identically.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            (batch_size, seq_len, d_model)
        """
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))
