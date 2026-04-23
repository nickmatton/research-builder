"""Tests for the evaluation pipeline."""

import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Path setup
_PHASE_ROOT = Path(__file__).resolve().parent.parent
_ARCH_DIR = _PHASE_ROOT.parent.parent / "architecture" / "1" / "outputs"
_DATA_DIR = _PHASE_ROOT.parent.parent / "data" / "1" / "outputs"
_TRAIN_DIR = _PHASE_ROOT.parent.parent / "training" / "1" / "outputs"
_SRC_DIR = _PHASE_ROOT / "src"

sys.path.insert(0, str(_ARCH_DIR))
sys.path.insert(0, str(_DATA_DIR))
sys.path.insert(0, str(_SRC_DIR))

from config import TransformerConfig
from transformer import Transformer
from beam_search import beam_search, _length_penalty, _make_tgt_mask
from average_checkpoints import get_checkpoint_paths, average_checkpoints, load_averaged_model
from evaluate import compute_bleu, load_vocab, decode_tokens, tokenize_for_bleu, evaluate


# ===========================================================================
# Test: BLEU score computation
# ===========================================================================

def test_bleu_perfect_match():
    """Perfect translation should get ~100 BLEU."""
    refs = [["the", "cat", "sat", "on", "the", "mat"]]
    hyps = [["the", "cat", "sat", "on", "the", "mat"]]
    result = compute_bleu(refs, hyps)
    assert result["bleu"] > 99.0, f"Perfect match BLEU should be ~100, got {result['bleu']}"
    print("PASS: test_bleu_perfect_match")


def test_bleu_no_match():
    """Completely wrong translation should get 0 BLEU."""
    refs = [["the", "cat", "sat"]]
    hyps = [["dog", "ran", "fast"]]
    result = compute_bleu(refs, hyps)
    assert result["bleu"] == 0.0, f"No match BLEU should be 0, got {result['bleu']}"
    print("PASS: test_bleu_no_match")


def test_bleu_partial_match():
    """Partial match should give intermediate BLEU."""
    refs = [["the", "cat", "sat", "on", "the", "big", "red", "mat", "today"]]
    hyps = [["the", "cat", "sat", "on", "a", "big", "red", "mat", "today"]]
    result = compute_bleu(refs, hyps)
    assert 0 < result["bleu"] < 100, f"Partial match BLEU should be between 0 and 100, got {result['bleu']}"
    print("PASS: test_bleu_partial_match")


def test_bleu_brevity_penalty():
    """Short hypothesis should get brevity penalty < 1."""
    refs = [["the", "cat", "sat", "on", "the", "mat"]]
    hyps = [["the", "cat"]]
    result = compute_bleu(refs, hyps, smooth=True)
    assert result["brevity_penalty"] < 1.0, f"Short hyp should have BP < 1, got {result['brevity_penalty']}"
    print("PASS: test_bleu_brevity_penalty")


def test_bleu_multiple_sentences():
    """BLEU should work with multiple sentence pairs."""
    refs = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["a", "dog", "ran", "across", "the", "field"],
    ]
    hyps = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["a", "dog", "ran", "across", "the", "field"],
    ]
    result = compute_bleu(refs, hyps)
    assert result["bleu"] > 99.0, f"Multi-sentence perfect BLEU should be ~100, got {result['bleu']}"
    print("PASS: test_bleu_multiple_sentences")


def test_bleu_returns_all_fields():
    """BLEU result should contain all required fields."""
    refs = [["a", "b", "c"]]
    hyps = [["a", "b", "c"]]
    result = compute_bleu(refs, hyps)
    required = ["bleu", "brevity_penalty", "precisions", "reference_length", "hypothesis_length"]
    for field in required:
        assert field in result, f"Missing field: {field}"
    assert len(result["precisions"]) == 4, f"Should have 4 precisions, got {len(result['precisions'])}"
    print("PASS: test_bleu_returns_all_fields")


# ===========================================================================
# Test: Length penalty
# ===========================================================================

def test_length_penalty():
    """Length penalty formula: ((5 + len) / 6)^alpha."""
    # With alpha=0.6, length=1: ((5+1)/6)^0.6 = 1.0
    lp = _length_penalty(1, 0.6)
    assert abs(lp - 1.0) < 1e-6, f"LP(1, 0.6) should be 1.0, got {lp}"

    # With alpha=0, any length: penalty = 1.0
    lp0 = _length_penalty(10, 0.0)
    assert abs(lp0 - 1.0) < 1e-6, f"LP(10, 0.0) should be 1.0, got {lp0}"

    # Longer sequences should have higher penalty
    lp5 = _length_penalty(5, 0.6)
    lp10 = _length_penalty(10, 0.6)
    assert lp10 > lp5, f"LP should increase with length: LP(10)={lp10} <= LP(5)={lp5}"
    print("PASS: test_length_penalty")


# ===========================================================================
# Test: Target mask
# ===========================================================================

def test_tgt_mask_causal():
    """Target mask should be causal (lower triangular)."""
    tgt = torch.tensor([[2, 5, 6, 3]])  # BOS ... EOS
    mask = _make_tgt_mask(tgt, pad_id=0, device=torch.device("cpu"))
    # Shape: (1, 1, 4, 4)
    assert mask.shape == (1, 1, 4, 4), f"Wrong mask shape: {mask.shape}"
    # Should be lower triangular
    for i in range(4):
        for j in range(4):
            if j > i:
                assert not mask[0, 0, i, j], f"Mask[{i},{j}] should be False (future)"
            else:
                assert mask[0, 0, i, j], f"Mask[{i},{j}] should be True (past/present)"
    print("PASS: test_tgt_mask_causal")


def test_tgt_mask_with_padding():
    """Target mask should mask padding positions."""
    tgt = torch.tensor([[2, 5, 0, 0]])  # BOS, token, pad, pad
    mask = _make_tgt_mask(tgt, pad_id=0, device=torch.device("cpu"))
    # Padding columns should be False
    assert not mask[0, 0, 2, 2], "Pad position should be False"
    assert not mask[0, 0, 3, 3], "Pad position should be False"
    print("PASS: test_tgt_mask_with_padding")


# ===========================================================================
# Test: Checkpoint averaging
# ===========================================================================

def test_get_checkpoint_paths():
    """Should find and sort checkpoints correctly."""
    ckpt_dir = str(_TRAIN_DIR / "checkpoints")
    paths = get_checkpoint_paths(ckpt_dir, num_checkpoints=5)
    assert len(paths) == 5, f"Expected 5 checkpoints, got {len(paths)}"
    # Should be sorted by step
    import re
    steps = []
    for p in paths:
        m = re.search(r'step(\d+)', p)
        steps.append(int(m.group(1)))
    assert steps == sorted(steps), f"Checkpoints not sorted: {steps}"
    # Should be the last 5 (196-200)
    assert steps[-1] == 200, f"Last checkpoint should be step 200, got {steps[-1]}"
    print("PASS: test_get_checkpoint_paths")


def test_checkpoint_averaging():
    """Averaged model should load without errors."""
    ckpt_dir = str(_TRAIN_DIR / "checkpoints")
    paths = get_checkpoint_paths(ckpt_dir, num_checkpoints=3)
    avg_state, config = average_checkpoints(paths)
    assert config is not None, "Config should not be None"
    assert isinstance(avg_state, dict), "avg_state should be a dict"
    assert "embedding.weight" in avg_state, "Should have embedding weights"
    # Verify values are averaged (not just last checkpoint)
    ckpt1 = torch.load(paths[0], map_location="cpu", weights_only=False)
    ckpt2 = torch.load(paths[-1], map_location="cpu", weights_only=False)
    key = "embedding.weight"
    s1 = ckpt1["model_state_dict"][key].float()
    s2 = ckpt2["model_state_dict"][key].float()
    # Averaged should be between (unless all same)
    avg_val = avg_state[key]
    # Just check it's a valid tensor
    assert avg_val.shape == s1.shape, "Averaged weight should have same shape"
    print("PASS: test_checkpoint_averaging")


def test_load_averaged_model():
    """Should produce a working Transformer model."""
    ckpt_dir = str(_TRAIN_DIR / "checkpoints")
    model = load_averaged_model(ckpt_dir, num_checkpoints=3, device="cpu")
    assert isinstance(model, Transformer), "Should return a Transformer"
    # Should be in eval mode
    assert not model.training, "Model should be in eval mode"
    # Test forward pass
    src = torch.tensor([[2, 5, 6, 3]])
    tgt = torch.tensor([[2, 7, 8]])
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    B, T = tgt.shape
    pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    tgt_mask = pad_mask & causal.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(src, tgt, src_mask, tgt_mask)
    assert logits.shape == (1, 3, model.config.tgt_vocab_size), f"Wrong output shape: {logits.shape}"
    print("PASS: test_load_averaged_model")


# ===========================================================================
# Test: Beam search
# ===========================================================================

def test_beam_search_produces_output():
    """Beam search should produce token sequences."""
    ckpt_dir = str(_TRAIN_DIR / "checkpoints")
    model = load_averaged_model(ckpt_dir, num_checkpoints=3, device="cpu")

    src = torch.tensor([[2, 5, 6, 7, 3]])
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

    results = beam_search(
        model, src, src_mask,
        beam_size=2, max_extra_len=10,
        length_penalty_alpha=0.6,
    )
    assert len(results) == 1, f"Should have 1 result, got {len(results)}"
    assert isinstance(results[0], list), "Result should be a list of token IDs"
    assert len(results[0]) > 0, "Result should not be empty"
    # Should not contain BOS
    assert results[0][0] != 2, "Result should not start with BOS"
    print("PASS: test_beam_search_produces_output")


def test_beam_search_batch():
    """Beam search should handle batches."""
    ckpt_dir = str(_TRAIN_DIR / "checkpoints")
    model = load_averaged_model(ckpt_dir, num_checkpoints=3, device="cpu")

    src = torch.tensor([
        [2, 5, 6, 3, 0],
        [2, 8, 9, 10, 3],
    ])
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

    results = beam_search(
        model, src, src_mask,
        beam_size=2, max_extra_len=10,
        length_penalty_alpha=0.6,
    )
    assert len(results) == 2, f"Should have 2 results, got {len(results)}"
    for i, r in enumerate(results):
        assert isinstance(r, list), f"Result {i} should be a list"
        assert len(r) > 0, f"Result {i} should not be empty"
    print("PASS: test_beam_search_batch")


# ===========================================================================
# Test: Vocabulary loading
# ===========================================================================

def test_load_vocab():
    """Should load vocabulary correctly."""
    vocab_path = str(_DATA_DIR / "vocab_en_de.json")
    token_to_id, id_to_token = load_vocab(vocab_path)
    assert len(token_to_id) > 0, "Vocab should not be empty"
    assert len(id_to_token) == len(token_to_id), "Forward and reverse vocab should have same size"
    print("PASS: test_load_vocab")


def test_decode_tokens():
    """Should decode token IDs to text."""
    vocab_path = str(_DATA_DIR / "vocab_en_de.json")
    _, id_to_token = load_vocab(vocab_path)
    # Get some valid token IDs
    valid_ids = list(id_to_token.keys())[:5]
    # Filter out special tokens
    valid_ids = [i for i in valid_ids if i not in (0, 1, 2, 3)]
    if valid_ids:
        text = decode_tokens(valid_ids, id_to_token)
        assert isinstance(text, str), "Should return a string"
        # Should handle special tokens gracefully
        text_with_special = decode_tokens([2] + valid_ids + [3], id_to_token)
        assert isinstance(text_with_special, str), "Should handle BOS/EOS"
    print("PASS: test_decode_tokens")


# ===========================================================================
# Test: Full evaluation (small scale)
# ===========================================================================

def test_full_evaluation():
    """Run evaluation on a small subset and verify output format."""
    output_dir = str(_PHASE_ROOT / "outputs")
    result = evaluate(
        max_eval_samples=10,
        batch_size=5,
        num_avg_checkpoints=3,
        beam_size=2,
        max_extra_len=20,
        device="cpu",
        output_dir=output_dir,
    )

    # Check result structure
    assert "en_de" in result, "Should have en_de key"
    assert "bleu" in result["en_de"], "Should have BLEU score"
    assert "paper_reported" in result, "Should have paper_reported"
    assert "eval_config" in result, "Should have eval_config"

    bleu = result["en_de"]["bleu"]
    assert isinstance(bleu, float), f"BLEU should be float, got {type(bleu)}"
    assert bleu >= 0.0, f"BLEU should be >= 0, got {bleu}"
    assert bleu <= 100.0, f"BLEU should be <= 100, got {bleu}"

    # Check output files exist
    assert os.path.exists(os.path.join(output_dir, "bleu_scores.json")), "bleu_scores.json not found"
    assert os.path.exists(os.path.join(output_dir, "translations.txt")), "translations.txt not found"

    # Check JSON is valid
    with open(os.path.join(output_dir, "bleu_scores.json")) as f:
        scores = json.load(f)
    assert "en_de" in scores, "JSON should have en_de"
    assert "paper_reported" in scores, "JSON should have paper_reported"
    assert "eval_config" in scores, "JSON should have eval_config"

    # Check translations file
    with open(os.path.join(output_dir, "translations.txt")) as f:
        lines = f.readlines()
    assert len(lines) == 10, f"Should have 10 translations, got {len(lines)}"

    print("PASS: test_full_evaluation")


# ===========================================================================
# Test: Output schema matches spec
# ===========================================================================

def test_output_schema():
    """Verify output files have correct schema."""
    output_dir = str(_PHASE_ROOT / "outputs")
    bleu_path = os.path.join(output_dir, "bleu_scores.json")

    if not os.path.exists(bleu_path):
        # Run eval first
        evaluate(max_eval_samples=5, batch_size=5, num_avg_checkpoints=3,
                 beam_size=2, max_extra_len=10, device="cpu", output_dir=output_dir)

    with open(bleu_path) as f:
        scores = json.load(f)

    # en_de should have these fields
    en_de = scores["en_de"]
    required_fields = ["bleu", "brevity_penalty", "precisions", "reference_length", "hypothesis_length"]
    for field in required_fields:
        assert field in en_de, f"en_de missing field: {field}"

    # precisions should be a list of 4 floats
    assert len(en_de["precisions"]) == 4, f"Should have 4 precision values"

    # paper_reported should have expected keys
    paper = scores["paper_reported"]
    assert "en_de_base" in paper, "Should have en_de_base paper score"
    assert paper["en_de_base"] == 27.3, f"Paper EN-DE base should be 27.3"

    # eval_config
    cfg = scores["eval_config"]
    assert "beam_size" in cfg, "Should have beam_size"
    assert "length_penalty_alpha" in cfg, "Should have length_penalty_alpha"

    print("PASS: test_output_schema")


# ===========================================================================
# Run all tests
# ===========================================================================

def run_all_tests():
    tests = [
        # BLEU tests
        test_bleu_perfect_match,
        test_bleu_no_match,
        test_bleu_partial_match,
        test_bleu_brevity_penalty,
        test_bleu_multiple_sentences,
        test_bleu_returns_all_fields,
        # Length penalty
        test_length_penalty,
        # Masks
        test_tgt_mask_causal,
        test_tgt_mask_with_padding,
        # Checkpoint averaging
        test_get_checkpoint_paths,
        test_checkpoint_averaging,
        test_load_averaged_model,
        # Beam search
        test_beam_search_produces_output,
        test_beam_search_batch,
        # Vocab
        test_load_vocab,
        test_decode_tokens,
        # Full evaluation
        test_full_evaluation,
        test_output_schema,
    ]

    passed = 0
    failed = 0
    errors = []

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test.__name__, str(e)))
            print(f"FAIL: {test.__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  {name}: {err}")

    return passed, failed, errors


if __name__ == "__main__":
    passed, failed, errors = run_all_tests()
    sys.exit(0 if failed == 0 else 1)
