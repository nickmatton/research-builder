"""Tests for the evaluation pipeline."""

import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load_model, evaluate, main


BASE = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = str(BASE.parents[1] / "training" / "1" / "outputs" / "checkpoint.pt")
TEST_DATASET_PATH = str(BASE.parents[1] / "data" / "1" / "outputs" / "test_dataset.pt")
METRICS_PATH = str(BASE / "outputs" / "metrics.json")


def test_load_model():
    """Model loads from checkpoint without errors."""
    model, ckpt = load_model(CHECKPOINT_PATH)
    assert model is not None
    assert ckpt["model_config"]["input_dim"] == 10
    assert ckpt["model_config"]["num_classes"] == 2
    # Model should be in eval mode
    assert not model.training


def test_evaluate_runs():
    """Evaluation produces metrics dict with required keys."""
    model, _ = load_model(CHECKPOINT_PATH)
    dataset = torch.load(TEST_DATASET_PATH, weights_only=False)
    metrics = evaluate(model, dataset)

    required_keys = ["test_accuracy", "test_loss", "num_test_samples", "num_correct", "paper_target_accuracy"]
    for k in required_keys:
        assert k in metrics, f"Missing key: {k}"


def test_accuracy_in_plausible_range():
    """Test accuracy is not zero, not absurdly large, and near the paper target."""
    model, _ = load_model(CHECKPOINT_PATH)
    dataset = torch.load(TEST_DATASET_PATH, weights_only=False)
    metrics = evaluate(model, dataset)

    acc = metrics["test_accuracy"]
    assert 0 < acc <= 100, f"Accuracy out of range: {acc}"
    # Should be within 5 percentage points of 95.2%
    assert abs(acc - 95.2) < 5.0, f"Accuracy {acc}% too far from paper target 95.2%"


def test_loss_is_reasonable():
    """Test loss should be a small positive number."""
    model, _ = load_model(CHECKPOINT_PATH)
    dataset = torch.load(TEST_DATASET_PATH, weights_only=False)
    metrics = evaluate(model, dataset)

    loss = metrics["test_loss"]
    assert 0 < loss < 10, f"Loss out of expected range: {loss}"


def test_num_samples_correct():
    """Number of test samples should match dataset size."""
    model, _ = load_model(CHECKPOINT_PATH)
    dataset = torch.load(TEST_DATASET_PATH, weights_only=False)
    metrics = evaluate(model, dataset)
    assert metrics["num_test_samples"] == len(dataset)


def test_output_file_exists_and_valid_json():
    """metrics.json should exist and be valid JSON with required schema."""
    # Run main to generate the file
    main()
    assert os.path.exists(METRICS_PATH), f"Output file not found: {METRICS_PATH}"
    with open(METRICS_PATH) as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert "test_accuracy" in data
    assert "paper_target_accuracy" in data


def test_deterministic():
    """Two evaluations should give the same result."""
    model, _ = load_model(CHECKPOINT_PATH)
    dataset = torch.load(TEST_DATASET_PATH, weights_only=False)
    m1 = evaluate(model, dataset)
    m2 = evaluate(model, dataset)
    assert m1["test_accuracy"] == m2["test_accuracy"]
    assert m1["test_loss"] == m2["test_loss"]


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"PASS: {name}")
            except AssertionError as e:
                print(f"FAIL: {name} - {e}")
            except Exception as e:
                print(f"ERROR: {name} - {e}")
