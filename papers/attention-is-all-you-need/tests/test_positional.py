"""Unit tests for src/positional.py — verifies the paper's PE formula."""

from __future__ import annotations

import math

import torch

from src.positional import PositionalEncoding


def test_pe_at_zero():
    """PE(0, 0) = sin(0) = 0; PE(0, 1) = cos(0) = 1."""
    pe = PositionalEncoding(d_model=8, max_len=10)
    # pe.pe is (1, max_len, d_model)
    assert torch.isclose(pe.pe[0, 0, 0], torch.tensor(0.0), atol=1e-7)
    assert torch.isclose(pe.pe[0, 0, 1], torch.tensor(1.0), atol=1e-7)


def test_pe_formula_matches_paper_at_sample_points():
    """Spot-check the formula at (pos=3, i=0): sin(3 / 10000^(0/8)) = sin(3)."""
    d_model = 8
    pe = PositionalEncoding(d_model=d_model, max_len=10)
    # PE(pos=3, 2*0) = sin(3 / 10000^0) = sin(3)
    assert torch.isclose(pe.pe[0, 3, 0], torch.tensor(math.sin(3.0)), atol=1e-6)
    # PE(pos=3, 2*0+1) = cos(3 / 10000^0) = cos(3)
    assert torch.isclose(pe.pe[0, 3, 1], torch.tensor(math.cos(3.0)), atol=1e-6)


def test_pe_addition_preserves_shape():
    pe = PositionalEncoding(d_model=16, max_len=100)
    x = torch.randn(3, 7, 16)
    out = pe(x)
    assert out.shape == x.shape


def test_pe_is_not_a_parameter():
    """PE is a buffer — not trained, but moves with .to(device)."""
    pe = PositionalEncoding(d_model=8, max_len=10)
    assert not any(p.requires_grad for p in pe.parameters())
    assert "pe" in dict(pe.named_buffers())
