"""Tests for src/wmt.py — token-budget batching + tokenize_pairs.

No network calls — uses an inline corpus rather than HF datasets. Loader
``load_wmt14_en_de`` is exercised in CI via the smoke run, not here.
"""

from __future__ import annotations

import pytest
import torch

from src import tokenize as tk
from src import wmt


@pytest.fixture(scope="module")
def tokenizer():
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "der schnelle braune fuchs springt über den faulen hund",
        "machine translation is fundamental",
        "maschinelle übersetzung ist grundlegend",
    ] * 50
    return tk.train_bpe(corpus, vocab_size=200)


def test_tokenize_pairs_drops_too_long(tokenizer):
    pairs = [
        {"en": "short", "de": "kurz"},
        {"en": "a " * 100, "de": "b " * 5},  # too long src
        {"en": "ok one", "de": "ok zwei"},
    ]
    out = wmt.tokenize_pairs(pairs, tokenizer, max_len=20)
    assert len(out) == 2  # dropped the long one


def test_tokenize_pairs_directions(tokenizer):
    pairs = [{"en": "hello world", "de": "hallo welt"}]
    en_de = wmt.tokenize_pairs(pairs, tokenizer, direction="en-de")
    de_en = wmt.tokenize_pairs(pairs, tokenizer, direction="de-en")
    assert len(en_de) == 1 and len(de_en) == 1
    # Source/target swap.
    assert en_de[0][0] != de_en[0][0] or en_de[0][1] != de_en[0][1]


def test_make_batch_shapes(tokenizer):
    pairs_ids = [
        ([1, 2, 3], [4, 5]),
        ([6, 7, 8, 9], [10, 11, 12]),
    ]
    batch = wmt.make_batch(pairs_ids)
    assert batch.src.shape == (2, 4)         # padded to longest (4)
    assert batch.tgt_in.shape == (2, 4)      # BOS + max(2, 3) = 4
    assert batch.tgt_out.shape == (2, 4)     # max(2, 3) + EOS = 4
    assert batch.tgt_in[0, 0] == tk.BOS_ID
    # tgt_out[i, last-non-pad] should be EOS for the longest target
    assert (batch.tgt_out == tk.EOS_ID).any()
    # Padding present where shorter
    assert (batch.src[0, 3] == tk.PAD_ID)


def test_token_budget_batches_respects_budget(tokenizer):
    pairs_ids = [([1] * (i + 3), [2] * (i + 3)) for i in range(30)]
    batches = list(wmt.token_budget_batches(pairs_ids, token_budget=80, seed=0))
    assert len(batches) > 1
    for b in batches:
        n, src_max = b.src.shape
        assert n * src_max <= 80, f"src budget exceeded: {n} × {src_max} = {n * src_max}"
        n2, tgt_max = b.tgt_in.shape
        assert n2 * tgt_max <= 80


def test_token_budget_batches_covers_all_pairs(tokenizer):
    pairs_ids = [([1, 2, 3, 4, 5], [6, 7, 8, 9])] * 12
    batches = list(wmt.token_budget_batches(pairs_ids, token_budget=80, seed=0))
    total = sum(b.src.size(0) for b in batches)
    assert total == 12, f"lost pairs: {total}/12"


def test_dtype_long(tokenizer):
    pairs_ids = [([1, 2], [3, 4])]
    batch = wmt.make_batch(pairs_ids)
    assert batch.src.dtype == torch.long
    assert batch.tgt_in.dtype == torch.long
    assert batch.tgt_out.dtype == torch.long
