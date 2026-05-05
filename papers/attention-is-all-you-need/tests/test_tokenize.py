"""Tests for src/tokenize.py — BPE training, encode/decode round-trip."""

from __future__ import annotations

from src import tokenize as tk


SAMPLE_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "der schnelle braune fuchs springt über den faulen hund",
    "machine translation is a fundamental task",
    "maschinelle übersetzung ist eine grundlegende aufgabe",
    "transformers replace recurrence with attention",
    "transformer ersetzen rekurrenz durch aufmerksamkeit",
] * 50


def test_train_bpe_basic():
    tok = tk.train_bpe(SAMPLE_CORPUS, vocab_size=200)
    assert tk.vocab_size(tok) > 0
    # Specials are always present.
    for special, expected_id in [(tk.PAD_TOKEN, tk.PAD_ID), (tk.BOS_TOKEN, tk.BOS_ID),
                                  (tk.EOS_TOKEN, tk.EOS_ID), (tk.UNK_TOKEN, tk.UNK_ID)]:
        ids = tok.encode(special).ids
        assert expected_id in ids, f"{special} missing from vocab"


def test_encode_decode_roundtrip():
    tok = tk.train_bpe(SAMPLE_CORPUS, vocab_size=200)
    text = "the quick brown fox"
    ids = tk.encode(tok, text)
    decoded = tk.decode(tok, ids)
    assert decoded.lower().replace(" ", "") == text.lower().replace(" ", ""), \
        f"roundtrip mismatch: {decoded!r} vs {text!r}"


def test_bos_eos_addition():
    tok = tk.train_bpe(SAMPLE_CORPUS, vocab_size=200)
    ids_plain = tk.encode(tok, "hello")
    ids_bos = tk.encode(tok, "hello", add_bos=True)
    ids_eos = tk.encode(tok, "hello", add_eos=True)
    ids_both = tk.encode(tok, "hello", add_bos=True, add_eos=True)
    assert ids_bos[0] == tk.BOS_ID
    assert ids_eos[-1] == tk.EOS_ID
    assert ids_both[0] == tk.BOS_ID and ids_both[-1] == tk.EOS_ID
    assert len(ids_both) == len(ids_plain) + 2


def test_decode_skips_specials():
    tok = tk.train_bpe(SAMPLE_CORPUS, vocab_size=200)
    ids_with_specials = [tk.BOS_ID] + tk.encode(tok, "hello") + [tk.EOS_ID, tk.PAD_ID]
    decoded = tk.decode(tok, ids_with_specials, skip_special=True)
    # No specials in output.
    assert tk.PAD_TOKEN not in decoded
    assert tk.BOS_TOKEN not in decoded
    assert tk.EOS_TOKEN not in decoded
