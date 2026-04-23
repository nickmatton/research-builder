"""
Tests for the training loop implementation.

Tests cover:
1. LR schedule matches spec at key steps
2. Label smoothing loss correctness
3. Loss decreases over first N steps (not diverging)
4. No NaN/Inf in gradients or loss
5. Checkpoints are written and loadable
6. Checkpoint averaging works
"""

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

# Path setup
_OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
_ARCH_DIR = Path(__file__).resolve().parent.parent.parent.parent / "architecture" / "1" / "outputs"
_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "1" / "outputs"

sys.path.insert(0, str(_OUTPUTS))
sys.path.insert(0, str(_ARCH_DIR))
sys.path.insert(0, str(_DATA_DIR))

from lr_scheduler import transformer_lr_lambda, get_transformer_scheduler
from train import (
    LabelSmoothingLoss,
    make_src_mask,
    make_tgt_mask,
    save_checkpoint,
    load_checkpoint,
    average_checkpoints,
    train,
)
from config import TransformerConfig
from transformer import Transformer


def test_lr_schedule_formula():
    """Test that LR schedule matches the paper formula at key steps."""
    d_model = 512
    warmup_steps = 4000

    # Formula: d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
    def expected_lr(step):
        step = max(step, 1)
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

    # Test at various steps
    test_steps = [1, 100, 1000, 4000, 4001, 8000, 50000, 100000]
    for step in test_steps:
        actual = transformer_lr_lambda(step, d_model, warmup_steps)
        expected = expected_lr(step)
        assert abs(actual - expected) < 1e-12, (
            f"LR mismatch at step {step}: got {actual}, expected {expected}"
        )

    print("PASS: test_lr_schedule_formula")


def test_lr_warmup_then_decay():
    """LR increases during warmup and decreases after."""
    d_model = 512
    warmup_steps = 4000

    # Warmup: LR should increase
    lr_10 = transformer_lr_lambda(10, d_model, warmup_steps)
    lr_100 = transformer_lr_lambda(100, d_model, warmup_steps)
    lr_2000 = transformer_lr_lambda(2000, d_model, warmup_steps)
    lr_4000 = transformer_lr_lambda(4000, d_model, warmup_steps)
    assert lr_10 < lr_100 < lr_2000 <= lr_4000, "LR should increase during warmup"

    # Decay: LR should decrease after warmup
    lr_5000 = transformer_lr_lambda(5000, d_model, warmup_steps)
    lr_10000 = transformer_lr_lambda(10000, d_model, warmup_steps)
    lr_50000 = transformer_lr_lambda(50000, d_model, warmup_steps)
    assert lr_4000 >= lr_5000 > lr_10000 > lr_50000, "LR should decay after warmup"

    print("PASS: test_lr_warmup_then_decay")


def test_lr_peak_at_warmup():
    """Peak LR should be at warmup_steps."""
    d_model = 512
    warmup_steps = 4000

    lr_peak = transformer_lr_lambda(warmup_steps, d_model, warmup_steps)
    # Check neighbors
    lr_before = transformer_lr_lambda(warmup_steps - 1, d_model, warmup_steps)
    lr_after = transformer_lr_lambda(warmup_steps + 1, d_model, warmup_steps)
    assert lr_peak >= lr_before, "Peak should be >= step before"
    assert lr_peak >= lr_after, "Peak should be >= step after"

    # Expected peak value: d_model^{-0.5} * warmup^{-0.5}
    expected_peak = (d_model ** -0.5) * (warmup_steps ** -0.5)
    assert abs(lr_peak - expected_peak) < 1e-10, f"Peak LR mismatch: {lr_peak} vs {expected_peak}"

    print("PASS: test_lr_peak_at_warmup")


def test_lr_scheduler_with_optimizer():
    """Test that the scheduler works with a real optimizer."""
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = get_transformer_scheduler(optimizer, d_model=512, warmup_steps=4000)

    lrs = []
    for step in range(1, 101):
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])

    # Should be increasing during warmup
    assert lrs[-1] > lrs[0], "LR should increase during early warmup"
    print("PASS: test_lr_scheduler_with_optimizer")


def test_label_smoothing_loss():
    """Test label smoothing loss basic properties."""
    vocab_size = 100
    criterion = LabelSmoothingLoss(vocab_size=vocab_size, padding_idx=0, smoothing=0.1)

    # Create dummy logits and targets
    batch = 4
    logits = torch.randn(batch, vocab_size)
    target = torch.tensor([5, 10, 0, 20])  # index 2 is pad

    loss = criterion(logits, target)
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive"
    assert math.isfinite(loss.item()), "Loss should be finite"

    print("PASS: test_label_smoothing_loss")


def test_label_smoothing_ignores_padding():
    """Padding tokens should not contribute to loss."""
    vocab_size = 50
    criterion = LabelSmoothingLoss(vocab_size=vocab_size, padding_idx=0, smoothing=0.1)

    logits = torch.randn(3, vocab_size)
    # All padding
    target_all_pad = torch.zeros(3, dtype=torch.long)
    loss_all_pad = criterion(logits, target_all_pad)
    assert loss_all_pad.item() == 0.0, f"Loss should be 0 for all-pad target, got {loss_all_pad.item()}"

    print("PASS: test_label_smoothing_ignores_padding")


def test_label_smoothing_confident_prediction():
    """Loss should be lower when model is confident on correct token."""
    vocab_size = 50
    criterion = LabelSmoothingLoss(vocab_size=vocab_size, padding_idx=0, smoothing=0.1)

    target = torch.tensor([5])

    # Very confident on correct answer
    logits_good = torch.full((1, vocab_size), -10.0)
    logits_good[0, 5] = 10.0

    # Uniform / bad prediction
    logits_bad = torch.zeros(1, vocab_size)

    loss_good = criterion(logits_good, target)
    loss_bad = criterion(logits_bad, target)
    assert loss_good.item() < loss_bad.item(), "Confident correct prediction should have lower loss"

    print("PASS: test_label_smoothing_confident_prediction")


def test_masks():
    """Test source and target mask shapes and properties."""
    src = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
    tgt = torch.tensor([[1, 2, 3, 0], [1, 2, 0, 0]])

    src_mask = make_src_mask(src)
    tgt_mask = make_tgt_mask(tgt)

    assert src_mask.shape == (2, 1, 1, 5), f"Bad src_mask shape: {src_mask.shape}"
    assert tgt_mask.shape == (2, 1, 4, 4), f"Bad tgt_mask shape: {tgt_mask.shape}"

    # Check src mask values
    assert src_mask[0, 0, 0, 0].item() == True  # non-pad
    assert src_mask[0, 0, 0, 3].item() == False  # pad

    # Check tgt causal: position 0 can only see position 0
    assert tgt_mask[0, 0, 0, 0].item() == True
    assert tgt_mask[0, 0, 0, 1].item() == False  # future

    print("PASS: test_masks")


def test_training_loss_decreases():
    """Run a short training loop and verify loss decreases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "checkpoints")
        log_path = os.path.join(tmpdir, "training_log.json")
        data_path = str(_DATA_DIR / "en_de_tokenized.pt")

        training_log = train(
            config_name="base",
            data_path=data_path,
            max_steps=50,
            tokens_per_batch=2000,  # small batches for speed
            checkpoint_interval_minutes=0.01,  # checkpoint quickly
            checkpoint_dir=ckpt_dir,
            log_path=log_path,
            device="cpu",
            warmup_steps=4000,
            label_smoothing=0.1,
            log_interval=1,  # log every step
        )

        assert len(training_log) > 0, "Training log is empty"

        # Check no NaN/Inf
        for entry in training_log:
            assert math.isfinite(entry["loss"]), f"Non-finite loss at step {entry['step']}: {entry['loss']}"
            assert math.isfinite(entry["lr"]), f"Non-finite LR at step {entry['step']}"

        # Check loss decreases: compare average of first 5 vs last 5
        first_losses = [e["loss"] for e in training_log[:5]]
        last_losses = [e["loss"] for e in training_log[-5:]]
        avg_first = sum(first_losses) / len(first_losses)
        avg_last = sum(last_losses) / len(last_losses)
        assert avg_last < avg_first, (
            f"Loss did not decrease: first 5 avg={avg_first:.4f}, last 5 avg={avg_last:.4f}"
        )

        print(f"  Loss decreased: {avg_first:.4f} -> {avg_last:.4f}")
        print("PASS: test_training_loss_decreases")


def test_checkpoint_save_load():
    """Test that checkpoints can be saved and loaded."""
    vocab_size = 100
    cfg = TransformerConfig.base(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size)
    model = Transformer(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = get_transformer_scheduler(optimizer, d_model=cfg.d_model, warmup_steps=4000)

    # Take a step to change state
    dummy_src = torch.randint(1, vocab_size, (2, 10))
    dummy_tgt = torch.randint(1, vocab_size, (2, 8))
    src_mask = make_src_mask(dummy_src)
    tgt_mask = make_tgt_mask(dummy_tgt[:, :-1])
    logits = model(dummy_src, dummy_tgt[:, :-1], src_mask, tgt_mask)
    loss = logits.sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
        save_checkpoint(model, optimizer, scheduler, step=42, epoch=1, path=ckpt_path)

        assert os.path.exists(ckpt_path), "Checkpoint file not created"

        # Load into fresh model
        model2 = Transformer(cfg)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler2 = get_transformer_scheduler(optimizer2, d_model=cfg.d_model, warmup_steps=4000)

        step, epoch = load_checkpoint(ckpt_path, model2, optimizer2, scheduler2)
        assert step == 42, f"Step mismatch: {step}"
        assert epoch == 1, f"Epoch mismatch: {epoch}"

        # Check weights match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.allclose(p1, p2), f"Weight mismatch for {n1}"

    print("PASS: test_checkpoint_save_load")


def test_checkpoint_averaging():
    """Test checkpoint averaging produces correct averaged weights."""
    vocab_size = 50
    cfg = TransformerConfig.base(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 3 checkpoints with known weights
        paths = []
        states = []
        for i in range(3):
            model = Transformer(cfg)
            # Set all params to i+1 for easy checking
            with torch.no_grad():
                for p in model.parameters():
                    p.fill_(float(i + 1))
            path = os.path.join(tmpdir, f"ckpt_{i}.pt")
            save_checkpoint(model, None, None, step=i, epoch=0, path=path)
            paths.append(path)

        # Average: (1+2+3)/3 = 2.0
        avg_model = Transformer(cfg)
        average_checkpoints(paths, avg_model)

        for name, p in avg_model.named_parameters():
            expected = 2.0
            assert torch.allclose(p, torch.full_like(p, expected), atol=1e-5), (
                f"Averaged weight mismatch for {name}: got {p.mean().item()}, expected {expected}"
            )

    print("PASS: test_checkpoint_averaging")


def test_training_produces_checkpoint_files():
    """Test that training actually writes checkpoint files and log."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "checkpoints")
        log_path = os.path.join(tmpdir, "training_log.json")
        data_path = str(_DATA_DIR / "en_de_tokenized.pt")

        train(
            config_name="base",
            data_path=data_path,
            max_steps=10,
            tokens_per_batch=2000,
            checkpoint_interval_minutes=0,  # checkpoint every step
            checkpoint_dir=ckpt_dir,
            log_path=log_path,
            device="cpu",
            log_interval=1,
        )

        # Check checkpoint exists
        ckpt_files = list(Path(ckpt_dir).glob("*.pt"))
        assert len(ckpt_files) >= 1, f"No checkpoint files found in {ckpt_dir}"

        # Check log exists and is valid JSON
        assert os.path.exists(log_path), "Training log not created"
        with open(log_path) as f:
            log = json.load(f)
        assert len(log) > 0, "Training log is empty"
        assert "step" in log[0], "Log entry missing 'step'"
        assert "loss" in log[0], "Log entry missing 'loss'"
        assert "lr" in log[0], "Log entry missing 'lr'"

    print("PASS: test_training_produces_checkpoint_files")


def test_no_nan_gradients():
    """Verify no NaN/Inf in gradients during training steps."""
    vocab_size = 100
    cfg = TransformerConfig.base(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size)
    model = Transformer(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = get_transformer_scheduler(optimizer, d_model=cfg.d_model, warmup_steps=4000)
    criterion = LabelSmoothingLoss(vocab_size=vocab_size, padding_idx=0, smoothing=0.1)

    for step in range(5):
        src = torch.randint(1, vocab_size, (4, 15))
        tgt = torch.randint(1, vocab_size, (4, 12))
        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt[:, :-1])

        logits = model(src, tgt[:, :-1], src_mask, tgt_mask)
        loss = criterion(logits.reshape(-1, vocab_size), tgt[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name} at step {step}"

        optimizer.step()
        scheduler.step()

    print("PASS: test_no_nan_gradients")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_lr_schedule_formula,
        test_lr_warmup_then_decay,
        test_lr_peak_at_warmup,
        test_lr_scheduler_with_optimizer,
        test_label_smoothing_loss,
        test_label_smoothing_ignores_padding,
        test_label_smoothing_confident_prediction,
        test_masks,
        test_no_nan_gradients,
        test_checkpoint_save_load,
        test_checkpoint_averaging,
        test_training_produces_checkpoint_files,
        test_training_loss_decreases,
    ]

    passed = 0
    failed = 0
    results = []

    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            passed += 1
            results.append((name, "passed", ""))
        except Exception as e:
            failed += 1
            results.append((name, "failed", str(e)))
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    for name, status, msg in results:
        mark = "✓" if status == "passed" else "✗"
        line = f"  {mark} {name}"
        if msg:
            line += f" — {msg}"
        print(line)
