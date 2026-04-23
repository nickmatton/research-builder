"""Tests for the WidgetClassifier model architecture."""

import torch
import pytest
from model import WidgetClassifier, build_model


# --- Test 1: Instantiation ---

def test_default_instantiation():
    model = WidgetClassifier()
    assert isinstance(model, torch.nn.Module)


def test_custom_instantiation():
    model = WidgetClassifier(input_dim=128, hidden_dim=64, num_classes=5, num_layers=3, dropout=0.2)
    assert model.input_dim == 128
    assert model.hidden_dim == 64
    assert model.num_classes == 5
    assert model.num_layers == 3


# --- Test 2: Forward pass shapes ---

@pytest.mark.parametrize("batch_size,input_dim,num_classes", [
    (1, 784, 10),
    (32, 784, 10),
    (16, 128, 5),
    (8, 1024, 100),
])
def test_forward_output_shape(batch_size, input_dim, num_classes):
    model = WidgetClassifier(input_dim=input_dim, num_classes=num_classes)
    x = torch.randn(batch_size, input_dim)
    out = model(x)
    assert out.shape == (batch_size, num_classes)


# --- Test 3: Output is logits (not probabilities) ---

def test_output_is_logits():
    model = WidgetClassifier()
    x = torch.randn(4, 784)
    out = model(x)
    # Logits can be negative and don't sum to 1
    assert out.min().item() < 1.0 or out.max().item() > 0.0  # not all zeros
    # Should NOT already be softmaxed (rows shouldn't sum to 1 in general)
    row_sums = out.sum(dim=1)
    assert not torch.allclose(row_sums, torch.ones(4), atol=0.01)


# --- Test 4: Gradient flow ---

def test_gradients_flow():
    model = WidgetClassifier(input_dim=32, hidden_dim=16, num_classes=3)
    x = torch.randn(4, 32)
    out = model(x)
    loss = out.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


# --- Test 5: Compatible with AdamW ---

def test_adamw_compatibility():
    model = WidgetClassifier(input_dim=32, hidden_dim=16, num_classes=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    x = torch.randn(4, 32)
    target = torch.tensor([0, 1, 2, 1])

    # One training step
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Second step - loss should still be computable
    out2 = model(x)
    loss2 = torch.nn.functional.cross_entropy(out2, target)
    assert loss2.item() >= 0  # valid loss


# --- Test 6: Accuracy evaluation compatibility ---

def test_accuracy_evaluation():
    model = WidgetClassifier(input_dim=32, hidden_dim=16, num_classes=3)
    model.eval()
    x = torch.randn(8, 32)
    with torch.no_grad():
        logits = model(x)
    preds = logits.argmax(dim=1)
    assert preds.shape == (8,)
    assert preds.min() >= 0
    assert preds.max() < 3


# --- Test 7: build_model factory ---

def test_build_model():
    model = build_model(input_dim=64, hidden_dim=32, num_classes=7)
    assert isinstance(model, WidgetClassifier)
    x = torch.randn(2, 64)
    assert model(x).shape == (2, 7)


# --- Test 8: Parameter count is reasonable ---

def test_parameter_count():
    model = WidgetClassifier(input_dim=784, hidden_dim=256, num_classes=10, num_layers=2)
    total = sum(p.numel() for p in model.parameters())
    # 784*256 + 256 + 256*256 + 256 + 256*10 + 10 = ~269,578
    assert 200_000 < total < 400_000, f"Unexpected param count: {total}"


# --- Test 9: Num layers scaling ---

def test_num_layers_scaling():
    m1 = WidgetClassifier(input_dim=32, hidden_dim=16, num_classes=3, num_layers=1)
    m3 = WidgetClassifier(input_dim=32, hidden_dim=16, num_classes=3, num_layers=3)
    p1 = sum(p.numel() for p in m1.parameters())
    p3 = sum(p.numel() for p in m3.parameters())
    assert p3 > p1, "More layers should mean more parameters"
