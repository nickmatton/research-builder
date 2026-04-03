"""Tests for the training loop."""

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

# Setup paths
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "architecture" / "1" / "outputs"))

from train import train
from model import WidgetClassifier


def make_synthetic_dataset(n=200, input_dim=10, num_classes=2):
    """Create a linearly-separable synthetic dataset for fast testing."""
    torch.manual_seed(42)
    X = torch.randn(n, input_dim)
    w = torch.randn(input_dim)
    y = (X @ w > 0).long()
    return TensorDataset(X, y)


def test_loss_decreases():
    """Training loss should decrease over the first epochs (not diverge)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ds_path = os.path.join(tmpdir, "ds.pt")
        torch.save(make_synthetic_dataset(), ds_path)
        out_dir = os.path.join(tmpdir, "out")

        log = train(dataset_path=ds_path, output_dir=out_dir, epochs=20, batch_size=32)

        first_loss = log[0]["train_loss"]
        last_loss = log[-1]["train_loss"]
        assert last_loss < first_loss, f"Loss did not decrease: {first_loss:.4f} -> {last_loss:.4f}"
        print("PASS: test_loss_decreases")


def test_no_nan_inf():
    """No NaN or Inf should appear in loss values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ds_path = os.path.join(tmpdir, "ds.pt")
        torch.save(make_synthetic_dataset(), ds_path)
        out_dir = os.path.join(tmpdir, "out")

        log = train(dataset_path=ds_path, output_dir=out_dir, epochs=10, batch_size=32)

        for entry in log:
            assert math.isfinite(entry["train_loss"]), f"Non-finite loss at epoch {entry['epoch']}"
        print("PASS: test_no_nan_inf")


def test_checkpoint_loadable():
    """Checkpoint should be saveable and loadable, and model should produce valid outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ds_path = os.path.join(tmpdir, "ds.pt")
        torch.save(make_synthetic_dataset(), ds_path)
        out_dir = os.path.join(tmpdir, "out")

        train(dataset_path=ds_path, output_dir=out_dir, epochs=5, batch_size=32)

        ckpt_path = os.path.join(out_dir, "checkpoint.pt")
        assert os.path.exists(ckpt_path), "Checkpoint file not found"

        ckpt = torch.load(ckpt_path, weights_only=False)
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "epoch" in ckpt
        assert "model_config" in ckpt

        # Load into a fresh model
        cfg = ckpt["model_config"]
        model = WidgetClassifier(input_dim=cfg["input_dim"], num_classes=cfg["num_classes"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        x = torch.randn(4, cfg["input_dim"])
        out = model(x)
        assert out.shape == (4, cfg["num_classes"])
        assert torch.isfinite(out).all()
        print("PASS: test_checkpoint_loadable")


def test_training_log_written():
    """training_log.json should exist and have correct structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ds_path = os.path.join(tmpdir, "ds.pt")
        torch.save(make_synthetic_dataset(), ds_path)
        out_dir = os.path.join(tmpdir, "out")

        train(dataset_path=ds_path, output_dir=out_dir, epochs=5, batch_size=32)

        log_path = os.path.join(out_dir, "training_log.json")
        assert os.path.exists(log_path), "Training log not found"

        with open(log_path) as f:
            log = json.load(f)

        assert len(log) == 5
        for entry in log:
            assert "epoch" in entry
            assert "train_loss" in entry
            assert "train_accuracy" in entry
            assert "lr" in entry
        print("PASS: test_training_log_written")


def test_constant_lr():
    """Learning rate should remain constant (no scheduler) as per spec."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ds_path = os.path.join(tmpdir, "ds.pt")
        torch.save(make_synthetic_dataset(), ds_path)
        out_dir = os.path.join(tmpdir, "out")

        log = train(dataset_path=ds_path, output_dir=out_dir, epochs=10, batch_size=32, lr=0.001)

        for entry in log:
            assert entry["lr"] == 0.001, f"LR changed at epoch {entry['epoch']}: {entry['lr']}"
        print("PASS: test_constant_lr")


if __name__ == "__main__":
    test_loss_decreases()
    test_no_nan_inf()
    test_checkpoint_loadable()
    test_training_log_written()
    test_constant_lr()
    print("\nAll tests passed!")
