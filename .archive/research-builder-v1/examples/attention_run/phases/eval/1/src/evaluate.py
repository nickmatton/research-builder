"""
Evaluation script for the Transformer model (Vaswani et al., 2017).

Implements the full evaluation pipeline:
1. Load and average checkpoints (last 5 for base, last 20 for big)
2. Run beam search decoding (beam_size=4, α=0.6)
3. Compute BLEU scores on newstest2014
4. Save translations and scores

Usage:
    python evaluate.py [--checkpoint_dir PATH] [--data_path PATH] [--vocab_path PATH]
"""

import argparse
import collections
import json
import math
import os
import sys
from pathlib import Path

import torch

# Path setup
_PHASE_ROOT = Path(__file__).resolve().parent.parent  # phases/eval/1
_ARCH_DIR = _PHASE_ROOT.parent.parent / "architecture" / "1" / "outputs"
_DATA_DIR = _PHASE_ROOT.parent.parent / "data" / "1" / "outputs"
_TRAIN_DIR = _PHASE_ROOT.parent.parent / "training" / "1" / "outputs"
_SRC_DIR = _PHASE_ROOT / "src"

sys.path.insert(0, str(_ARCH_DIR))
sys.path.insert(0, str(_DATA_DIR))
sys.path.insert(0, str(_SRC_DIR))

from config import TransformerConfig
from transformer import Transformer
from beam_search import beam_search
from average_checkpoints import load_averaged_model, get_checkpoint_paths, average_checkpoints


# ---- BLEU Score Implementation ----

def compute_bleu(
    reference_corpus: list[list[str]],
    hypothesis_corpus: list[list[str]],
    max_order: int = 4,
    smooth: bool = False,
) -> dict:
    """Compute BLEU score (based on the standard multi-bleu implementation).

    Args:
        reference_corpus: List of reference token lists.
        hypothesis_corpus: List of hypothesis token lists.
        max_order: Maximum n-gram order (default: 4 for BLEU-4).
        smooth: Whether to apply smoothing (add-1).

    Returns:
        Dictionary with BLEU score and component details.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    hypothesis_length = 0

    for references, hypothesis in zip(reference_corpus, hypothesis_corpus):
        reference_length += len(references)
        hypothesis_length += len(hypothesis)

        # Count n-grams in reference
        ref_ngram_counts = _get_ngrams(references, max_order)
        # Count n-grams in hypothesis
        hyp_ngram_counts = _get_ngrams(hypothesis, max_order)

        # Clipped counts
        overlap = {
            ngram: min(count, ref_ngram_counts.get(ngram, 0))
            for ngram, count in hyp_ngram_counts.items()
        }

        for ngram, count in overlap.items():
            matches_by_order[len(ngram) - 1] += count

        for order in range(1, max_order + 1):
            possible = max(len(hypothesis) - order + 1, 0)
            possible_matches_by_order[order - 1] += possible

    precisions = []
    for i in range(max_order):
        if smooth:
            precisions.append(
                (matches_by_order[i] + 1.0) / (possible_matches_by_order[i] + 1.0)
            )
        else:
            if possible_matches_by_order[i] > 0:
                precisions.append(
                    matches_by_order[i] / possible_matches_by_order[i]
                )
            else:
                precisions.append(0.0)

    # Geometric mean of precisions
    if min(precisions) > 0:
        p_log_sum = sum(math.log(p) for p in precisions) / max_order
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0.0

    # Brevity penalty
    if hypothesis_length > reference_length:
        bp = 1.0
    elif hypothesis_length == 0:
        bp = 0.0
    else:
        bp = math.exp(1.0 - reference_length / hypothesis_length)

    bleu = geo_mean * bp * 100.0  # As percentage

    return {
        "bleu": bleu,
        "brevity_penalty": bp,
        "precisions": [p * 100.0 for p in precisions],
        "reference_length": reference_length,
        "hypothesis_length": hypothesis_length,
        "matches_by_order": matches_by_order,
        "possible_matches_by_order": possible_matches_by_order,
    }


def _get_ngrams(tokens: list[str], max_order: int) -> dict[tuple, int]:
    """Extract n-gram counts from token list."""
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(len(tokens) - order + 1):
            ngram = tuple(tokens[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


# ---- Vocabulary / Detokenization ----

def load_vocab(vocab_path: str) -> tuple[dict[str, int], dict[int, str]]:
    """Load vocabulary from JSON file.

    Returns:
        (token_to_id, id_to_token) dictionaries.
    """
    with open(vocab_path, "r") as f:
        token_to_id = json.load(f)
    id_to_token = {v: k for k, v in token_to_id.items()}
    return token_to_id, id_to_token


def decode_tokens(token_ids: list[int], id_to_token: dict[int, str],
                  bos_id: int = 2, eos_id: int = 3, pad_id: int = 0) -> str:
    """Convert token IDs back to text.

    Handles BPE-style tokens (Ġ prefix = space before token).
    Strips BOS/EOS/PAD tokens.
    """
    tokens = []
    for tid in token_ids:
        if tid in (bos_id, eos_id, pad_id):
            continue
        token = id_to_token.get(tid, "<unk>")
        tokens.append(token)

    # Join BPE tokens
    text = "".join(tokens)
    # BPE uses Ġ for space
    text = text.replace("Ġ", " ").strip()
    return text


def tokenize_for_bleu(text: str) -> list[str]:
    """Simple whitespace tokenization for BLEU computation."""
    return text.split()


# ---- Main Evaluation ----

def make_src_mask(src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """(B, 1, 1, S) – True where token is NOT pad."""
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def evaluate(
    checkpoint_dir: str | None = None,
    data_path: str | None = None,
    vocab_path: str | None = None,
    output_dir: str | None = None,
    num_avg_checkpoints: int = 5,
    beam_size: int = 4,
    length_penalty_alpha: float = 0.6,
    max_extra_len: int = 50,
    batch_size: int = 32,
    device: str | None = None,
    max_eval_samples: int | None = None,
) -> dict:
    """Run full evaluation pipeline.

    Args:
        checkpoint_dir: Path to checkpoint directory.
        data_path: Path to tokenized test data.
        vocab_path: Path to vocabulary JSON.
        output_dir: Directory for output files.
        num_avg_checkpoints: Number of checkpoints to average.
        beam_size: Beam search width.
        length_penalty_alpha: Length penalty α.
        max_extra_len: Max output = input_len + this.
        batch_size: Batch size for inference.
        device: Device string.
        max_eval_samples: Limit evaluation to N samples (for testing).

    Returns:
        Dictionary with BLEU scores and metadata.
    """
    # Defaults
    if checkpoint_dir is None:
        checkpoint_dir = str(_TRAIN_DIR / "checkpoints")
    if data_path is None:
        data_path = str(_DATA_DIR / "en_de_tokenized.pt")
    if vocab_path is None:
        vocab_path = str(_DATA_DIR / "vocab_en_de.json")
    if output_dir is None:
        output_dir = str(_PHASE_ROOT / "outputs")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Data path: {data_path}")
    print(f"Vocab path: {vocab_path}")

    # 1. Load averaged model
    print("\n=== Loading averaged model ===")
    model = load_averaged_model(checkpoint_dir, num_avg_checkpoints, device=device)
    model.eval()
    print(f"Model config: {model.config}")

    # 2. Load vocabulary
    print("\n=== Loading vocabulary ===")
    token_to_id, id_to_token = load_vocab(vocab_path)
    print(f"Vocabulary size: {len(token_to_id)}")

    # 3. Load test data
    print("\n=== Loading test data ===")
    data = torch.load(data_path, weights_only=False)
    src_sequences = data["src"]
    tgt_sequences = data["tgt"]
    metadata = data.get("metadata", {})
    print(f"Test pairs: {len(src_sequences)}")
    print(f"Metadata: {metadata}")

    if max_eval_samples is not None:
        src_sequences = src_sequences[:max_eval_samples]
        tgt_sequences = tgt_sequences[:max_eval_samples]
        print(f"Limited to {max_eval_samples} samples")

    # 4. Run beam search decoding
    print(f"\n=== Running beam search (beam={beam_size}, α={length_penalty_alpha}) ===")
    all_hypotheses = []
    all_references = []
    translations = []

    num_batches = (len(src_sequences) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(src_sequences))

        # Pad source sequences in batch
        batch_src = src_sequences[start:end]
        batch_tgt = tgt_sequences[start:end]

        max_src_len = max(len(s) for s in batch_src)
        src_tensor = torch.zeros(len(batch_src), max_src_len, dtype=torch.long, device=device)
        for i, s in enumerate(batch_src):
            src_tensor[i, :len(s)] = torch.tensor(s, dtype=torch.long)

        src_mask = make_src_mask(src_tensor)

        # Decode
        decoded_seqs = beam_search(
            model, src_tensor, src_mask,
            beam_size=beam_size,
            max_extra_len=max_extra_len,
            length_penalty_alpha=length_penalty_alpha,
            device=device,
        )

        # Convert to text and collect for BLEU
        for i, (hyp_ids, ref_ids) in enumerate(zip(decoded_seqs, batch_tgt)):
            hyp_text = decode_tokens(hyp_ids, id_to_token)
            ref_text = decode_tokens(ref_ids, id_to_token)

            hyp_tokens = tokenize_for_bleu(hyp_text)
            ref_tokens = tokenize_for_bleu(ref_text)

            all_hypotheses.append(hyp_tokens)
            all_references.append(ref_tokens)
            translations.append(hyp_text)

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            print(f"  Decoded batch {batch_idx + 1}/{num_batches}")

    # 5. Compute BLEU
    print("\n=== Computing BLEU ===")
    bleu_result = compute_bleu(all_references, all_hypotheses)
    print(f"BLEU score: {bleu_result['bleu']:.2f}")
    print(f"Brevity penalty: {bleu_result['brevity_penalty']:.4f}")
    print(f"Precisions: {[f'{p:.1f}' for p in bleu_result['precisions']]}")

    # Paper-reported scores for comparison
    paper_scores = {
        "en_de_base": 27.3,
        "en_de_big": 28.4,
        "en_fr_base": 38.1,
        "en_fr_big": 41.8,
    }

    # 6. Save results
    bleu_scores = {
        "en_de": {
            "bleu": bleu_result["bleu"],
            "brevity_penalty": bleu_result["brevity_penalty"],
            "precisions": bleu_result["precisions"],
            "reference_length": bleu_result["reference_length"],
            "hypothesis_length": bleu_result["hypothesis_length"],
        },
        "paper_reported": paper_scores,
        "eval_config": {
            "beam_size": beam_size,
            "length_penalty_alpha": length_penalty_alpha,
            "max_extra_len": max_extra_len,
            "num_avg_checkpoints": num_avg_checkpoints,
            "num_eval_samples": len(src_sequences),
        },
    }

    # Save BLEU scores
    bleu_path = os.path.join(output_dir, "bleu_scores.json")
    with open(bleu_path, "w") as f:
        json.dump(bleu_scores, f, indent=2)
    print(f"\nBLEU scores saved to {bleu_path}")

    # Save translations
    trans_path = os.path.join(output_dir, "translations.txt")
    with open(trans_path, "w") as f:
        for t in translations:
            f.write(t + "\n")
    print(f"Translations saved to {trans_path}")

    return bleu_scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate Transformer")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--vocab_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_avg_checkpoints", type=int, default=5)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--length_penalty_alpha", type=float, default=0.6)
    parser.add_argument("--max_extra_len", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    args = parser.parse_args()

    evaluate(
        checkpoint_dir=args.checkpoint_dir,
        data_path=args.data_path,
        vocab_path=args.vocab_path,
        output_dir=args.output_dir,
        num_avg_checkpoints=args.num_avg_checkpoints,
        beam_size=args.beam_size,
        length_penalty_alpha=args.length_penalty_alpha,
        max_extra_len=args.max_extra_len,
        batch_size=args.batch_size,
        device=args.device,
        max_eval_samples=args.max_eval_samples,
    )


if __name__ == "__main__":
    main()
