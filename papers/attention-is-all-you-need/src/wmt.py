"""WMT 2014 EN-DE data loader.

Paper §5.1 (p.7):
    "We trained on the standard WMT 2014 English-German dataset consisting of
    about 4.5 million sentence pairs. Sentences were encoded using byte-pair
    encoding, which has a shared source-target vocabulary of about 37000 tokens.
    ... Sentence pairs were batched together by approximate sequence length.
    Each training batch contained a set of sentence pairs containing
    approximately 25000 source tokens and 25000 target tokens."

We load via HuggingFace ``datasets`` (which mirrors the standard preprocessed
WMT 2014 release) and emit token-budget batches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from tokenizers import Tokenizer

from . import tokenize as tk


# Default per-batch token budget; paper says ~25k src + ~25k tgt (§5.1).
DEFAULT_TOKEN_BUDGET = 25000


@dataclass
class Batch:
    src: torch.Tensor          # (B, L_src) padded with PAD_ID
    tgt_in: torch.Tensor       # (B, L_tgt+1) BOS + tgt
    tgt_out: torch.Tensor      # (B, L_tgt+1) tgt + EOS


def load_wmt14_en_de(split: str = "train", limit: int | None = None) -> list[dict]:
    """Load WMT 2014 EN-DE pairs from HuggingFace datasets.

    Returns list of {"en": str, "de": str}. ``limit`` truncates to N examples
    (used for smoke runs / tests; None loads the full split).
    """
    from datasets import load_dataset  # imported lazily — heavy
    ds = load_dataset("wmt14", "de-en", split=split, streaming=limit is not None and limit > 100000)
    out: list[dict] = []
    for i, item in enumerate(ds):
        if limit is not None and i >= limit:
            break
        pair = item["translation"]
        out.append({"en": pair["en"], "de": pair["de"]})
    return out


def train_tokenizer_from_pairs(
    pairs: list[dict],
    vocab_size: int = 37000,
    save_path: Path | None = None,
) -> Tokenizer:
    """Train shared EN+DE BPE on a list of pairs."""
    def iter_text() -> Iterator[str]:
        for p in pairs:
            yield p["en"]
            yield p["de"]
    tok = tk.train_bpe(iter_text(), vocab_size=vocab_size)
    if save_path is not None:
        tk.save(tok, save_path)
    return tok


def tokenize_pairs(
    pairs: list[dict],
    tok: Tokenizer,
    direction: str = "en-de",
    max_len: int = 256,
) -> list[tuple[list[int], list[int]]]:
    """Return list of (src_ids, tgt_ids) WITHOUT BOS/EOS — those are added at batch time.

    ``direction`` is "en-de" or "de-en". ``max_len`` drops pairs whose either
    side exceeds this token count (long sentences hurt token-budget batching).
    """
    if direction == "en-de":
        src_lang, tgt_lang = "en", "de"
    elif direction == "de-en":
        src_lang, tgt_lang = "de", "en"
    else:
        raise ValueError(f"unknown direction: {direction}")

    out: list[tuple[list[int], list[int]]] = []
    for p in pairs:
        s = tk.encode(tok, p[src_lang])
        t = tk.encode(tok, p[tgt_lang])
        if len(s) == 0 or len(t) == 0:
            continue
        if len(s) > max_len or len(t) > max_len:
            continue
        out.append((s, t))
    return out


def _pad_to(ids: list[int], length: int) -> list[int]:
    return ids + [tk.PAD_ID] * (length - len(ids))


def make_batch(pairs: list[tuple[list[int], list[int]]]) -> Batch:
    """Build a padded Batch from a list of (src, tgt) ID sequences."""
    src_max = max(len(s) for s, _ in pairs)
    tgt_max = max(len(t) for _, t in pairs)

    src = torch.tensor([_pad_to(s, src_max) for s, _ in pairs], dtype=torch.long)
    tgt_in = torch.tensor([_pad_to([tk.BOS_ID] + t, tgt_max + 1) for _, t in pairs],
                          dtype=torch.long)
    tgt_out = torch.tensor([_pad_to(t + [tk.EOS_ID], tgt_max + 1) for _, t in pairs],
                           dtype=torch.long)
    return Batch(src=src, tgt_in=tgt_in, tgt_out=tgt_out)


def token_budget_batches(
    pairs: list[tuple[list[int], list[int]]],
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    seed: int = 0,
) -> Iterator[Batch]:
    """Yield batches whose total padded token count ≤ token_budget.

    Strategy (paper §5.1, "batched together by approximate sequence length"):
      1. Bucket-sort by max(src_len, tgt_len). Sentences of similar length pack tighter.
      2. Walk the sorted list, greedily packing into batches that respect the
         budget. Padded length × batch_size must stay ≤ token_budget on EACH side.
      3. Lightly shuffle bucket order each epoch (else training data is sorted).
    """
    import random
    rng = random.Random(seed)

    indexed = list(enumerate(pairs))
    indexed.sort(key=lambda ix: max(len(ix[1][0]), len(ix[1][1])))

    batches: list[list[tuple[list[int], list[int]]]] = []
    current: list[tuple[list[int], list[int]]] = []
    cur_src_max = 0
    cur_tgt_max = 0

    def fits(src: list[int], tgt: list[int]) -> bool:
        new_src_max = max(cur_src_max, len(src))
        new_tgt_max = max(cur_tgt_max, len(tgt))
        n = len(current) + 1
        return n * new_src_max <= token_budget and n * (new_tgt_max + 1) <= token_budget

    for _, (s, t) in indexed:
        if not current or fits(s, t):
            current.append((s, t))
            cur_src_max = max(cur_src_max, len(s))
            cur_tgt_max = max(cur_tgt_max, len(t))
        else:
            batches.append(current)
            current = [(s, t)]
            cur_src_max = len(s)
            cur_tgt_max = len(t)
    if current:
        batches.append(current)

    rng.shuffle(batches)
    for batch in batches:
        yield make_batch(batch)
