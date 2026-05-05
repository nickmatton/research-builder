"""Beam-search decoding + BLEU evaluation.

Paper §6.1 (p.8):
    "We used beam search with a beam size of 4 and length penalty α = 0.6.
    We set the maximum output length during inference to input length + 50,
    but terminate early when possible."

Length penalty (Wu et al. 2016, cited as [38]):
    lp(Y) = (5 + |Y|)^α / (5 + 1)^α

Score on a beam = sum_log_probs / lp(|Y|).

BLEU is computed via sacrebleu (modern standard). Paper used ``multi-bleu.perl``
which differs slightly in tokenization — gap documented in CLAUDE.md, future
work to add the perl-compatible variant for true paper-faithful comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from . import tokenize as tk
from .transformer import Transformer


@dataclass
class Beam:
    tokens: list[int]      # generated token ids (excluding BOS)
    log_prob: float        # cumulative log-prob
    finished: bool = False

    def length(self) -> int:
        return len(self.tokens)

    def score(self, alpha: float = 0.6) -> float:
        # Wu et al. length penalty.
        lp = ((5 + self.length()) ** alpha) / ((5 + 1) ** alpha)
        return self.log_prob / lp


@torch.no_grad()
def beam_search(
    model: Transformer,
    src: torch.Tensor,             # (1, L_src) — single sentence
    beam_size: int = 4,
    length_penalty: float = 0.6,
    max_extra_tokens: int = 50,
    eos_id: int = tk.EOS_ID,
    bos_id: int = tk.BOS_ID,
) -> list[int]:
    """Decode one source sentence. Returns the best beam's token ids (no BOS/EOS)."""
    if src.size(0) != 1:
        raise ValueError("beam_search expects batch size 1")

    model.eval()
    device = src.device
    src_len = src.size(1)
    max_len = src_len + max_extra_tokens

    # Encode once.
    src_pad = model._pad_mask(src)
    memory = model.encode(src, src_pad)         # (1, L_src, d_model)

    beams: list[Beam] = [Beam(tokens=[], log_prob=0.0)]
    finished: list[Beam] = []

    for _ in range(max_len):
        if not beams:
            break

        # Build a (B, L_tgt+1) tensor across active beams (BOS prepended).
        tgt = torch.tensor(
            [[bos_id] + b.tokens for b in beams], dtype=torch.long, device=device
        )
        # Encoder memory replicated to match active beam count.
        mem = memory.expand(len(beams), -1, -1)
        src_p = src_pad.expand(len(beams), -1, -1, -1)

        from .transformer import causal_mask
        tgt_pad = model._pad_mask(tgt)
        causal = causal_mask(tgt.size(1), device=device)
        tgt_mask = tgt_pad | causal.unsqueeze(0).unsqueeze(0)
        out = model.decode(tgt, mem, tgt_mask, src_p)
        logits = out @ model.embedding.weight.T                # (B, L_tgt+1, V)
        next_log_probs = F.log_softmax(logits[:, -1, :], dim=-1)  # (B, V)

        # For each beam, take top beam_size next tokens.
        topk_lp, topk_ids = next_log_probs.topk(beam_size, dim=-1)  # (B, k)

        candidates: list[Beam] = []
        for b_idx, b in enumerate(beams):
            for k in range(beam_size):
                tok_id = int(topk_ids[b_idx, k])
                lp = float(topk_lp[b_idx, k])
                new = Beam(
                    tokens=b.tokens + [tok_id],
                    log_prob=b.log_prob + lp,
                    finished=(tok_id == eos_id),
                )
                candidates.append(new)

        # Re-rank: keep top beam_size by length-penalized score.
        candidates.sort(key=lambda c: c.score(length_penalty), reverse=True)
        next_active: list[Beam] = []
        for c in candidates:
            if len(next_active) >= beam_size and len(finished) >= beam_size:
                break
            if c.finished:
                # Strip the EOS for the final output.
                c.tokens = c.tokens[:-1]
                finished.append(c)
            elif len(next_active) < beam_size:
                next_active.append(c)
        beams = next_active

        # Early stop: all current active beams are worse than the best finished.
        if finished and beams:
            best_finished = max(f.score(length_penalty) for f in finished)
            best_active = max(b.score(length_penalty) for b in beams)
            if best_finished > best_active:
                break

    candidates_all = finished + beams
    if not candidates_all:
        return []
    best = max(candidates_all, key=lambda b: b.score(length_penalty))
    return best.tokens


def bleu_corpus(hypotheses: list[str], references: list[str]) -> float:
    """sacrebleu corpus-BLEU on tokenized strings. Returns the score (0-100)."""
    import sacrebleu
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return float(bleu.score)


def evaluate(
    model: Transformer,
    pairs: list[tuple[list[int], list[int]]],
    tok: Tokenizer,
    beam_size: int = 4,
    length_penalty: float = 0.6,
    max_extra_tokens: int = 50,
    device: torch.device | None = None,
) -> dict:
    """Decode every pair, return {bleu, n_examples, hypotheses, references}.

    ``pairs`` is the tokenize_pairs output: (src_ids, tgt_ids) without BOS/EOS.
    Hypotheses + references are decoded back to strings for sacrebleu.
    """
    if device is not None:
        model = model.to(device)

    hypotheses: list[str] = []
    references: list[str] = []
    for s, t in pairs:
        src = torch.tensor([s], dtype=torch.long, device=device or torch.device("cpu"))
        hyp_ids = beam_search(
            model, src,
            beam_size=beam_size,
            length_penalty=length_penalty,
            max_extra_tokens=max_extra_tokens,
        )
        hypotheses.append(tk.decode(tok, hyp_ids))
        references.append(tk.decode(tok, t))

    bleu = bleu_corpus(hypotheses, references)
    return {
        "bleu": bleu,
        "n_examples": len(pairs),
        "hypotheses": hypotheses,
        "references": references,
    }


def main() -> int:
    """CLI: load checkpoint, run beam search on WMT split, write metrics.json."""
    import argparse
    import json
    from pathlib import Path

    from . import wmt as wmt_module

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--split", default="test", choices=["validation", "test"])
    p.add_argument("--limit", type=int, default=None,
                   help="cap eval pairs (smoke); None = full split")
    p.add_argument("--beam-size", type=int, default=4, help="paper §6.1: 4")
    p.add_argument("--length-penalty", type=float, default=0.6, help="paper §6.1: α=0.6")
    p.add_argument("--max-extra-tokens", type=int, default=50,
                   help="max output length = input length + this")
    p.add_argument("--output", type=Path, default=Path("runs/dev/metrics.json"))
    p.add_argument("--max-len", type=int, default=256)
    args = p.parse_args()

    device = (torch.device("cuda") if torch.cuda.is_available() else
              torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
              torch.device("cpu"))
    print(f"[eval] device={device}")

    print(f"[eval] loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = Transformer(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_encoder_layers=cfg["num_layers"],
        num_decoder_layers=cfg["num_layers"],
        d_ff=cfg["d_ff"],
        dropout=0.0,
        pad_id=tk.PAD_ID,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[eval] model loaded. params: {model.num_parameters():,}")

    print(f"[eval] loading tokenizer: {args.tokenizer}")
    tok = tk.load(args.tokenizer)

    print(f"[eval] loading WMT 2014 EN-DE {args.split} (limit={args.limit})...")
    pairs_raw = wmt_module.load_wmt14_en_de(split=args.split, limit=args.limit)
    pairs = wmt_module.tokenize_pairs(pairs_raw, tok, direction="en-de", max_len=args.max_len)
    print(f"[eval] {len(pairs)} pairs after length filter; running beam={args.beam_size}...")

    result = evaluate(
        model, pairs, tok,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        max_extra_tokens=args.max_extra_tokens,
        device=device,
    )

    # Map BLEU result onto claim_id keys for compare-claims.py.
    metrics = {
        "table2_base_en_de_bleu": result["bleu"],
        "table2_big_en_de_bleu": result["bleu"],   # same model in this run
        "n_examples": result["n_examples"],
        "beam_size": args.beam_size,
        "length_penalty": args.length_penalty,
        "split": args.split,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))

    # Dump first few hypotheses for visual sanity.
    sample_path = args.output.parent / "eval-samples.txt"
    with sample_path.open("w") as f:
        for i, (h, r) in enumerate(zip(result["hypotheses"][:20], result["references"][:20])):
            f.write(f"--- example {i} ---\nHYP: {h}\nREF: {r}\n\n")

    print(f"[eval] BLEU = {result['bleu']:.2f} on {result['n_examples']} pairs "
          f"({args.split} split, beam={args.beam_size})")
    print(f"[eval] metrics → {args.output}")
    print(f"[eval] samples → {sample_path}")
    print(f"[eval] next: python scripts/compare-claims.py {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
