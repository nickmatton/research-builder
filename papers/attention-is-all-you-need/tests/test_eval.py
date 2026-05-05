"""Tests for src/eval.py — beam search basics + BLEU plumbing."""

from __future__ import annotations

import pytest
import torch

from src import tokenize as tk
from src.eval import Beam, beam_search, bleu_corpus
from src.transformer import Transformer


def test_beam_score_length_penalty_favors_longer_at_equal_log_prob():
    """Wu et al. 2016 length penalty: lp(Y) = (5+|Y|)^α / 6^α grows with length.

    score = log_prob / lp; with same negative log_prob, dividing by a larger
    lp brings it closer to zero — so the LONGER beam scores higher. This is
    the whole point: counteract beam search's preference for early-stopping."""
    short = Beam(tokens=[1, 2, 3], log_prob=-3.0)
    long = Beam(tokens=[1, 2, 3, 4, 5, 6, 7], log_prob=-3.0)
    assert long.score(alpha=0.6) > short.score(alpha=0.6)


def test_beam_score_alpha_zero_is_no_penalty():
    """alpha=0 collapses to raw log-prob (lp=1)."""
    beam = Beam(tokens=[1, 2, 3], log_prob=-3.0)
    assert beam.score(alpha=0.0) == pytest.approx(-3.0)


def test_beam_search_returns_valid_sequence():
    """Tiny model, beam=2: decoder produces a non-empty token list ≤ max_len."""
    torch.manual_seed(0)
    model = Transformer(vocab_size=20, d_model=16, num_heads=2,
                         num_encoder_layers=1, num_decoder_layers=1, d_ff=32)
    model.eval()
    src = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)
    out = beam_search(model, src, beam_size=2, max_extra_tokens=10)
    assert isinstance(out, list)
    assert len(out) <= src.size(1) + 10
    # A randomly-initialized model can emit any vocab id including specials;
    # we only assert the structural properties (list, bounded length).


def test_beam_search_handles_beam_one():
    """beam=1 = greedy; should still work (regression guard)."""
    torch.manual_seed(0)
    model = Transformer(vocab_size=20, d_model=16, num_heads=2,
                         num_encoder_layers=1, num_decoder_layers=1, d_ff=32)
    model.eval()
    src = torch.tensor([[3, 4, 5]], dtype=torch.long)
    out = beam_search(model, src, beam_size=1, max_extra_tokens=8)
    assert isinstance(out, list)


def test_bleu_corpus_perfect_score():
    """Identical hypotheses + references → BLEU 100."""
    hyps = ["the cat sat on the mat", "the dog ran fast"]
    refs = ["the cat sat on the mat", "the dog ran fast"]
    score = bleu_corpus(hyps, refs)
    assert score == pytest.approx(100.0, abs=0.1)


def test_bleu_corpus_zero_overlap():
    """Disjoint vocab → near-zero BLEU."""
    hyps = ["xxx yyy zzz", "aaa bbb ccc"]
    refs = ["the quick brown", "fox jumps high"]
    score = bleu_corpus(hyps, refs)
    assert score < 5.0


def test_bleu_corpus_partial():
    """Partial match — needs ≥4-gram overlap for sacrebleu BLEU > 0."""
    hyps = ["the cat sat on the floor today"]
    refs = ["the cat sat on the mat today"]
    score = bleu_corpus(hyps, refs)
    assert 0.0 < score < 100.0
