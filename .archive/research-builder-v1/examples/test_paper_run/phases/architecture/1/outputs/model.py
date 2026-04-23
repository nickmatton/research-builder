"""
Model architecture for Widget Performance study.

NOTE: The paper does not specify model architecture details. This implements
a flexible MLP classifier as a reasonable default, compatible with AdamW
training and accuracy evaluation as required by the paper's methods section.

The model is parameterized so that downstream phases can adjust capacity.
"""

import torch
import torch.nn as nn


class WidgetClassifier(nn.Module):
    """A multi-layer perceptron classifier.

    Args:
        input_dim: Dimensionality of input features.
        hidden_dim: Width of hidden layers.
        num_classes: Number of output classes.
        num_layers: Number of hidden layers (default 2).
        dropout: Dropout probability (default 0.1).
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        num_classes: int = 10,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers

        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        features = self.features(x)
        logits = self.classifier(features)
        return logits


def build_model(**kwargs) -> WidgetClassifier:
    """Factory function to create a WidgetClassifier with given config."""
    return WidgetClassifier(**kwargs)
