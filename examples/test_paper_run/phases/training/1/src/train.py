"""
Training loop for Widget Performance study.

Trains a WidgetClassifier using AdamW optimizer for 100 epochs
with the hyperparameters specified in Section 2 of the paper.
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add model definition to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "architecture" / "1" / "outputs"))
from model import WidgetClassifier


def train(
    dataset_path: str,
    output_dir: str,
    lr: float = 0.001,
    weight_decay: float = 0.01,
    epochs: int = 100,
    batch_size: int = 32,
    device: str = "cpu",
):
    """Run the full training loop.

    Args:
        dataset_path: Path to train_dataset.pt (a TensorDataset).
        output_dir: Directory to write checkpoint.pt and training_log.json.
        lr: Learning rate (default 0.001).
        weight_decay: Weight decay for AdamW (default 0.01).
        epochs: Number of training epochs (default 100).
        batch_size: Batch size (default 32).
        device: Device string.

    Returns:
        training_log: list of dicts with per-epoch metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    dataset = torch.load(dataset_path, weights_only=False)
    # Infer dimensions from data
    x_sample, y_sample = dataset[0]
    input_dim = x_sample.shape[0]
    num_classes = int(max(d[1] for d in dataset).item()) + 1

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Build model
    model = WidgetClassifier(input_dim=input_dim, num_classes=num_classes)
    model.to(device)

    # Optimizer & loss (paper: AdamW, lr=0.001, wd=0.01; loss unspecified -> CrossEntropy for classification)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    training_log = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        entry = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_accuracy": epoch_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        training_log.append(entry)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | loss={epoch_loss:.4f} | acc={epoch_acc:.4f}")

    # Save checkpoint
    checkpoint_path = os.path.join(output_dir, "checkpoint.pt")
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": training_log[-1]["train_loss"],
        "train_accuracy": training_log[-1]["train_accuracy"],
        "model_config": {
            "input_dim": input_dim,
            "num_classes": num_classes,
        },
    }, checkpoint_path)

    # Save training log
    log_path = os.path.join(output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"Checkpoint saved to {checkpoint_path}")
    print(f"Training log saved to {log_path}")

    return training_log


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    dataset_path = str(base.parents[1] / "data" / "1" / "outputs" / "train_dataset.pt")
    output_dir = str(base / "outputs")
    train(dataset_path=dataset_path, output_dir=output_dir)
