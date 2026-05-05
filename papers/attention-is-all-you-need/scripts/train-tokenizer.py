#!/usr/bin/env python3
"""Train the shared EN+DE BPE tokenizer (paper §5.1, p.7).

    python scripts/train-tokenizer.py [--vocab-size 37000] [--limit N]

Defaults match the paper: shared source-target vocab of ~37000 tokens. Use
``--limit`` for smoke runs (e.g. 10000 pairs) so this finishes in seconds
instead of minutes on the full 4.5M-pair WMT 2014 EN-DE corpus.

Output: ``data/wmt14_en_de/tokenizer.json``. Subsequent ``src.train --data wmt``
loads from this path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import wmt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, default=37000)
    p.add_argument("--limit", type=int, default=None,
                   help="cap WMT pairs used for training (smoke); None = full")
    p.add_argument("--output", type=Path, default=Path("data/wmt14_en_de/tokenizer.json"))
    args = p.parse_args()

    print(f"[tokenizer] loading WMT 2014 EN-DE train (limit={args.limit})...")
    pairs = wmt.load_wmt14_en_de(split="train", limit=args.limit)
    print(f"[tokenizer] training BPE on {len(pairs)} pairs ({2 * len(pairs)} sentences) "
          f"vocab={args.vocab_size}")
    tok = wmt.train_tokenizer_from_pairs(pairs, vocab_size=args.vocab_size, save_path=args.output)
    from src import tokenize as tk
    print(f"[tokenizer] saved → {args.output}  (vocab_size={tk.vocab_size(tok)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
