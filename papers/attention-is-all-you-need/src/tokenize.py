"""BPE tokenizer wrapper.

Paper §5.1 (p.7): "Sentences were encoded using byte-pair encoding [3], which
has a shared source-target vocabulary of about 37000 tokens."

We use HuggingFace ``tokenizers`` (the Rust-backed library, not the
``transformers`` package) for speed and simplicity. Trained tokenizers are
saved as a single ``tokenizer.json`` file under ``data/wmt14_en_de/``.

Special tokens (reserve ids 0-3):
    0  <pad>
    1  <bos>
    2  <eos>
    3  <unk>
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


def train_bpe(
    corpus: Iterable[str],
    vocab_size: int = 37000,
    min_frequency: int = 2,
) -> Tokenizer:
    """Train a shared-vocab BPE tokenizer on an iterable of strings.

    Pass an iterator of EN+DE sentences interleaved (so the vocab is shared
    across the language pair, per paper §5.1).
    """
    tok = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIALS,
    )
    tok.train_from_iterator(corpus, trainer)
    return tok


def save(tok: Tokenizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))


def load(path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def encode(tok: Tokenizer, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
    ids = tok.encode(text).ids
    if add_bos:
        ids = [BOS_ID] + ids
    if add_eos:
        ids = ids + [EOS_ID]
    return ids


def decode(tok: Tokenizer, ids: list[int], skip_special: bool = True) -> str:
    if skip_special:
        ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
    return tok.decode(ids)


def vocab_size(tok: Tokenizer) -> int:
    return tok.get_vocab_size()
