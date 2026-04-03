"""
Evaluation pipeline for Widget Performance study.

Loads a trained checkpoint and evaluates on the held-out test set.
Primary metric: accuracy (paper target: 95.2%).
"""

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add model definition to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "architecture" / "1" / "outputs"))
from model import WidgetClassifier


def load_model(checkpoint_path: str, device: str = "cpu") -> WidgetClassifier:
    """Load a WidgetClassifier from a training checkpoint."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config = checkpoint["model_config"]
    model = WidgetClassifier(
        input_dim=config["input_dim"],
        num_classes=config["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def evaluate(model: nn.Module, dataset, batch_size: int = 64, device: str = "cpu"):
    """Evaluate model on a TensorDataset, returning metrics dict."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    total_loss = 0.0
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            total_loss += loss.item() * batch_y.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch_y.cpu().tolist())

    accuracy = correct / total
    avg_loss = total_loss / total

    # Per-class accuracy
    num_classes = max(max(all_preds), max(all_labels)) + 1
    per_class = {}
    for c in range(num_classes):
        mask = [i for i, l in enumerate(all_labels) if l == c]
        if mask:
            class_correct = sum(1 for i in mask if all_preds[i] == c)
            per_class[f"class_{c}_accuracy"] = class_correct / len(mask)

    metrics = {
        "test_accuracy": round(accuracy * 100, 2),
        "test_loss": round(avg_loss, 6),
        "num_test_samples": total,
        "num_correct": correct,
        "paper_target_accuracy": 95.2,
        **per_class,
    }
    return metrics


def main():
    base = Path(__file__).resolve().parents[1]
    checkpoint_path = str(base.parents[1] / "training" / "1" / "outputs" / "checkpoint.pt")
    test_dataset_path = str(base.parents[1] / "data" / "1" / "outputs" / "test_dataset.pt")
    output_dir = str(base / "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading checkpoint from {checkpoint_path}")
    model, checkpoint = load_model(checkpoint_path)
    print(f"Training epoch: {checkpoint['epoch']}, train_acc: {checkpoint['train_accuracy']:.4f}")

    print(f"Loading test dataset from {test_dataset_path}")
    test_dataset = torch.load(test_dataset_path, weights_only=False)

    print("Running evaluation...")
    metrics = evaluate(model, test_dataset)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {metrics_path}")
    print(f"Test Accuracy: {metrics['test_accuracy']}% (paper target: {metrics['paper_target_accuracy']}%)")
    print(f"Test Loss: {metrics['test_loss']}")

    return metrics


if __name__ == "__main__":
    main()
