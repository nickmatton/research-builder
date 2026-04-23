"""Unit tests for src/transformer.py — forward shape, mask correctness, training basics."""

from __future__ import annotations

import torch

from src.data import PAD_ID, synthetic_batch
from src.transformer import Transformer, causal_mask


def test_forward_shape():
    model = Transformer(vocab_size=64, d_model=32, num_heads=4, num_encoder_layers=2,
                         num_decoder_layers=2, d_ff=64)
    src = torch.randint(3, 64, (2, 5))
    tgt = torch.randint(3, 64, (2, 6))
    logits = model(src, tgt)
    assert logits.shape == (2, 6, 64)


def test_param_count_positive():
    model = Transformer(vocab_size=64, d_model=32, num_heads=4, num_encoder_layers=2,
                         num_decoder_layers=2, d_ff=64)
    assert model.num_parameters() > 0


def test_loss_is_finite_on_dummy_input():
    model = Transformer(vocab_size=64, d_model=32, num_heads=4, num_encoder_layers=1,
                         num_decoder_layers=1, d_ff=64)
    src = torch.randint(3, 64, (2, 5))
    tgt = torch.randint(3, 64, (2, 6))
    logits = model(src, tgt)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)
    )
    assert torch.isfinite(loss)


def test_causal_mask_is_upper_triangular():
    m = causal_mask(4)
    # Position i can attend to all j ≤ i; mask is True for j > i.
    expected = torch.tensor([
        [False, True,  True,  True],
        [False, False, True,  True],
        [False, False, False, True],
        [False, False, False, False],
    ])
    assert torch.equal(m, expected)


def test_synthetic_batch_shapes():
    src, tgt_in, tgt_out = synthetic_batch(batch_size=3, src_len=4, tgt_len=5, vocab_size=20, seed=1)
    assert src.shape == (3, 4)
    assert tgt_in.shape == (3, 6)         # BOS + tgt_len
    assert tgt_out.shape == (3, 6)        # tgt_len + EOS
    assert (src >= 3).all() and (src < 20).all()


def test_gradients_flow_through_all_layers():
    """Every parameter that requires grad should receive a non-None gradient on backward."""
    model = Transformer(vocab_size=32, d_model=16, num_heads=2, num_encoder_layers=2,
                         num_decoder_layers=2, d_ff=32)
    src, tgt_in, tgt_out = synthetic_batch(batch_size=2, src_len=4, tgt_len=4, vocab_size=32, seed=0)
    logits = model(src, tgt_in)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 32), tgt_out.reshape(-1))
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"no gradient for {name}"
            # Layer norm bias / pad embedding row may be exactly 0, but non-pad
            # parameters should generally have nonzero gradient.
