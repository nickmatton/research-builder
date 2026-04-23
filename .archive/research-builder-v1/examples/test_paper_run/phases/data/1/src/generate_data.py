"""
Generate synthetic widget classification dataset.

The paper ("Test Paper: A Study of Widget Performance") does not specify the dataset.
Based on context clues (batch_size=32, classification accuracy metric, AdamW optimizer),
we generate a synthetic binary classification dataset with:
- 1000 training samples, 200 test samples
- 10 input features
- 2 classes (binary classification)
- Linearly separable with some noise for realistic difficulty

Output: train_dataset.pt and test_dataset.pt saved as TensorDataset objects.
"""

import torch
from torch.utils.data import TensorDataset
import os
import numpy as np


def generate_widget_dataset(
    n_train: int = 1000,
    n_test: int = 200,
    n_features: int = 10,
    n_classes: int = 2,
    noise_std: float = 0.3,
    seed: int = 42,
):
    """Generate a synthetic classification dataset for widget performance."""
    rng = np.random.RandomState(seed)

    n_total = n_train + n_test

    # Generate features from standard normal
    X = rng.randn(n_total, n_features).astype(np.float32)

    # Generate a random weight vector for linear decision boundary
    w = rng.randn(n_features).astype(np.float32)
    logits = X @ w + noise_std * rng.randn(n_total).astype(np.float32)

    # Binary labels from sign of logits
    y = (logits > 0).astype(np.int64)

    # Split
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Convert to tensors
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test),
    )

    return train_dataset, test_dataset


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    train_dataset, test_dataset = generate_widget_dataset()

    train_path = os.path.join(output_dir, "train_dataset.pt")
    test_path = os.path.join(output_dir, "test_dataset.pt")

    torch.save(train_dataset, train_path)
    torch.save(test_dataset, test_path)

    print(f"Train dataset: {len(train_dataset)} samples saved to {train_path}")
    print(f"Test dataset: {len(test_dataset)} samples saved to {test_path}")
    print(f"Features shape: {train_dataset.tensors[0].shape}")
    print(f"Labels shape: {train_dataset.tensors[1].shape}")
    print(f"Label distribution (train): {torch.bincount(train_dataset.tensors[1]).tolist()}")
    print(f"Label distribution (test): {torch.bincount(test_dataset.tensors[1]).tolist()}")


if __name__ == "__main__":
    main()
