"""
Tokenize the parallel data using trained tokenizers and save as .pt files.

Each .pt file contains a dict with:
- 'src': list of tokenized source sequences (list of list of int)
- 'tgt': list of tokenized target sequences (list of list of int)
- 'metadata': dict with dataset info
"""

import os
import torch
from tokenizers import Tokenizer


def tokenize_dataset(raw_path, tokenizer_path, output_path, dataset_name, full_size):
    """Tokenize a parallel dataset and save as .pt."""
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    print(f"Reading raw data from {raw_path}")
    src_tokens = []
    tgt_tokens = []
    num_pairs = 0

    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "\t" not in line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue

            src_enc = tokenizer.encode(parts[0])
            tgt_enc = tokenizer.encode(parts[1])

            src_tokens.append(src_enc.ids)
            tgt_tokens.append(tgt_enc.ids)
            num_pairs += 1

            if num_pairs % 10000 == 0:
                print(f"  Tokenized {num_pairs} pairs...")

    print(f"Total tokenized pairs: {num_pairs}")

    # Compute statistics
    src_lengths = [len(s) for s in src_tokens]
    tgt_lengths = [len(t) for t in tgt_tokens]

    metadata = {
        "dataset": dataset_name,
        "num_pairs": num_pairs,
        "full_intended_size": full_size,
        "avg_src_length": sum(src_lengths) / len(src_lengths) if src_lengths else 0,
        "avg_tgt_length": sum(tgt_lengths) / len(tgt_lengths) if tgt_lengths else 0,
        "max_src_length": max(src_lengths) if src_lengths else 0,
        "max_tgt_length": max(tgt_lengths) if tgt_lengths else 0,
        "vocab_size": tokenizer.get_vocab_size(),
    }

    data = {
        "src": src_tokens,
        "tgt": tgt_tokens,
        "metadata": metadata,
    }

    torch.save(data, output_path)
    print(f"Saved tokenized data to {output_path}")
    print(f"  Pairs: {num_pairs}")
    print(f"  Avg src length: {metadata['avg_src_length']:.1f}")
    print(f"  Avg tgt length: {metadata['avg_tgt_length']:.1f}")
    print(f"  Vocab size: {metadata['vocab_size']}")

    return data


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir = os.path.join(base_dir, "outputs")

    # EN-DE
    print("=" * 60)
    print("Tokenizing EN-DE dataset...")
    print("=" * 60)
    tokenize_dataset(
        raw_path=os.path.join(outputs_dir, "raw_en_de.txt"),
        tokenizer_path=os.path.join(outputs_dir, "tokenizer_en_de.json"),
        output_path=os.path.join(outputs_dir, "en_de_tokenized.pt"),
        dataset_name="WMT2014_EN_DE",
        full_size=4_500_000,
    )

    # EN-FR
    print("=" * 60)
    print("Tokenizing EN-FR dataset...")
    print("=" * 60)
    tokenize_dataset(
        raw_path=os.path.join(outputs_dir, "raw_en_fr.txt"),
        tokenizer_path=os.path.join(outputs_dir, "tokenizer_en_fr.json"),
        output_path=os.path.join(outputs_dir, "en_fr_tokenized.pt"),
        dataset_name="WMT2014_EN_FR",
        full_size=36_000_000,
    )

    print("\nTokenization complete!")


if __name__ == "__main__":
    main()
