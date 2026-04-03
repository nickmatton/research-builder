"""Tests for the generated widget dataset."""

import os
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from generate_data import generate_widget_dataset

BATCH_SIZE = 32
N_TRAIN = 1000
N_TEST = 200
N_FEATURES = 10
N_CLASSES = 2


def test_dataset_sizes():
    """Train and test datasets have correct number of samples."""
    train_ds, test_ds = generate_widget_dataset()
    assert len(train_ds) == N_TRAIN, f"Expected {N_TRAIN} train samples, got {len(train_ds)}"
    assert len(test_ds) == N_TEST, f"Expected {N_TEST} test samples, got {len(test_ds)}"


def test_feature_shape_and_dtype():
    """Features are float32 with correct dimensionality."""
    train_ds, test_ds = generate_widget_dataset()
    for name, ds in [("train", train_ds), ("test", test_ds)]:
        X, y = ds.tensors
        assert X.ndim == 2, f"{name} features should be 2D"
        assert X.shape[1] == N_FEATURES, f"{name} features dim should be {N_FEATURES}"
        assert X.dtype == torch.float32, f"{name} features should be float32"


def test_label_shape_and_dtype():
    """Labels are int64 with correct shape."""
    train_ds, test_ds = generate_widget_dataset()
    for name, ds in [("train", train_ds), ("test", test_ds)]:
        X, y = ds.tensors
        assert y.ndim == 1, f"{name} labels should be 1D"
        assert y.shape[0] == X.shape[0], f"{name} labels count should match features"
        assert y.dtype == torch.int64, f"{name} labels should be int64"


def test_no_nans():
    """No NaN values in features or labels."""
    train_ds, test_ds = generate_widget_dataset()
    for name, ds in [("train", train_ds), ("test", test_ds)]:
        X, y = ds.tensors
        assert not torch.isnan(X).any(), f"{name} features contain NaN"
        assert not torch.isnan(y.float()).any(), f"{name} labels contain NaN"


def test_label_values():
    """Labels are valid class indices (0 or 1 for binary)."""
    train_ds, test_ds = generate_widget_dataset()
    for name, ds in [("train", train_ds), ("test", test_ds)]:
        y = ds.tensors[1]
        unique = torch.unique(y)
        assert all(v in [0, 1] for v in unique), f"{name} labels should be 0 or 1, got {unique}"


def test_label_balance():
    """Labels are roughly balanced (neither class < 30% of samples)."""
    train_ds, _ = generate_widget_dataset()
    y = train_ds.tensors[1]
    counts = torch.bincount(y)
    min_ratio = counts.min().item() / len(y)
    assert min_ratio > 0.3, f"Label imbalance: min class ratio {min_ratio:.2f} < 0.3"


def test_dataloader_batching():
    """DataLoader produces batches of correct size."""
    train_ds, _ = generate_widget_dataset()
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    batch_X, batch_y = next(iter(loader))
    assert batch_X.shape == (BATCH_SIZE, N_FEATURES)
    assert batch_y.shape == (BATCH_SIZE,)


def test_reproducibility():
    """Same seed produces identical datasets."""
    t1, s1 = generate_widget_dataset(seed=42)
    t2, s2 = generate_widget_dataset(seed=42)
    assert torch.equal(t1.tensors[0], t2.tensors[0])
    assert torch.equal(t1.tensors[1], t2.tensors[1])


def test_saved_files():
    """Saved .pt files load correctly as TensorDataset."""
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    train_path = os.path.join(output_dir, "train_dataset.pt")
    test_path = os.path.join(output_dir, "test_dataset.pt")

    assert os.path.exists(train_path), f"Missing {train_path}"
    assert os.path.exists(test_path), f"Missing {test_path}"

    train_ds = torch.load(train_path, weights_only=False)
    test_ds = torch.load(test_path, weights_only=False)

    assert isinstance(train_ds, TensorDataset)
    assert isinstance(test_ds, TensorDataset)
    assert len(train_ds) == N_TRAIN
    assert len(test_ds) == N_TEST


ALL_TESTS = [
    ("test_dataset_sizes", test_dataset_sizes),
    ("test_feature_shape_and_dtype", test_feature_shape_and_dtype),
    ("test_label_shape_and_dtype", test_label_shape_and_dtype),
    ("test_no_nans", test_no_nans),
    ("test_label_values", test_label_values),
    ("test_label_balance", test_label_balance),
    ("test_dataloader_batching", test_dataloader_batching),
    ("test_reproducibility", test_reproducibility),
    ("test_saved_files", test_saved_files),
]


if __name__ == "__main__":
    passed = 0
    failed = 0
    results = []
    for name, fn in ALL_TESTS:
        try:
            fn()
            print(f"  PASS: {name}")
            passed += 1
            results.append((name, "passed", ""))
        except Exception as e:
            print(f"  FAIL: {name} - {e}")
            failed += 1
            results.append((name, "failed", str(e)))

    print(f"\n{passed}/{passed + failed} tests passed")
    if failed:
        sys.exit(1)
