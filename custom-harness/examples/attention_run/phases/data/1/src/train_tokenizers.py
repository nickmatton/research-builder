"""
Train BPE and WordPiece tokenizers for EN-DE and EN-FR respectively.

EN-DE: Byte-pair encoding (BPE) with shared source-target vocabulary of ~37,000 tokens
EN-FR: Word-piece vocabulary of 32,000 tokens

Uses the HuggingFace tokenizers library for fast training.
"""

import json
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, processors

# Special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


def read_parallel_data(filepath):
    """Read tab-separated parallel data, return (src_lines, tgt_lines)."""
    src_lines = []
    tgt_lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "\t" not in line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                src_lines.append(parts[0])
                tgt_lines.append(parts[1])
    return src_lines, tgt_lines


def train_bpe_tokenizer(texts, vocab_size=37000):
    """Train a BPE tokenizer on given texts."""
    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Add post-processing for BOS/EOS
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        pair=f"{BOS_TOKEN} $A {EOS_TOKEN} {BOS_TOKEN} $B {EOS_TOKEN}",
        special_tokens=[
            (BOS_TOKEN, tokenizer.token_to_id(BOS_TOKEN)),
            (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
        ],
    )

    return tokenizer


def train_wordpiece_tokenizer(texts, vocab_size=32000):
    """Train a WordPiece tokenizer on given texts."""
    tokenizer = Tokenizer(models.WordPiece(unk_token=UNK_TOKEN))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Add post-processing for BOS/EOS
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        pair=f"{BOS_TOKEN} $A {EOS_TOKEN} {BOS_TOKEN} $B {EOS_TOKEN}",
        special_tokens=[
            (BOS_TOKEN, tokenizer.token_to_id(BOS_TOKEN)),
            (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
        ],
    )

    return tokenizer


def save_vocab(tokenizer, path):
    """Save vocabulary as JSON: {token: id}."""
    vocab = tokenizer.get_vocab()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Saved vocabulary ({len(vocab)} tokens) to {path}")
    return vocab


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir = os.path.join(base_dir, "outputs")

    # --- EN-DE: BPE with shared vocab of ~37K ---
    print("=" * 60)
    print("Training BPE tokenizer for EN-DE (shared vocab ~37K)...")
    print("=" * 60)

    en_de_path = os.path.join(outputs_dir, "raw_en_de.txt")
    src_lines, tgt_lines = read_parallel_data(en_de_path)
    print(f"Loaded {len(src_lines)} EN-DE sentence pairs")

    # Shared vocabulary: train on both source and target
    all_en_de_texts = src_lines + tgt_lines
    bpe_tokenizer = train_bpe_tokenizer(all_en_de_texts, vocab_size=37000)

    vocab_en_de = save_vocab(bpe_tokenizer, os.path.join(outputs_dir, "vocab_en_de.json"))
    bpe_tokenizer.save(os.path.join(outputs_dir, "tokenizer_en_de.json"))

    # --- EN-FR: WordPiece with 32K vocab ---
    print("=" * 60)
    print("Training WordPiece tokenizer for EN-FR (32K vocab)...")
    print("=" * 60)

    en_fr_path = os.path.join(outputs_dir, "raw_en_fr.txt")
    src_lines_fr, tgt_lines_fr = read_parallel_data(en_fr_path)
    print(f"Loaded {len(src_lines_fr)} EN-FR sentence pairs")

    # Train on both source and target
    all_en_fr_texts = src_lines_fr + tgt_lines_fr
    wp_tokenizer = train_wordpiece_tokenizer(all_en_fr_texts, vocab_size=32000)

    vocab_en_fr = save_vocab(wp_tokenizer, os.path.join(outputs_dir, "vocab_en_fr.json"))
    wp_tokenizer.save(os.path.join(outputs_dir, "tokenizer_en_fr.json"))

    print("\nTokenizer training complete!")
    print(f"  EN-DE BPE vocab size: {len(vocab_en_de)}")
    print(f"  EN-FR WordPiece vocab size: {len(vocab_en_fr)}")


if __name__ == "__main__":
    main()
