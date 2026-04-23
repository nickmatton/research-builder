"""Data loaders.

For the smoke / overfit ladder rungs, ``synthetic_batch`` returns a fixed
deterministic batch of integer token IDs. No real WMT yet — that's Phase 2
of notes/plan.md and lives behind a HuggingFace ``datasets`` load.
"""

from __future__ import annotations

import torch


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


def synthetic_batch(
    batch_size: int = 4,
    src_len: int = 6,
    tgt_len: int = 6,
    vocab_size: int = 100,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic random batch for smoke + overfit-one-batch.

    Returns (src, tgt_in, tgt_out) where:
      - src: (B, L_src) token ids in [3, vocab_size)
      - tgt_in: (B, L_tgt) token ids prefixed with BOS
      - tgt_out: (B, L_tgt) token ids — what we predict (shifted left of tgt_in)

    PAD/BOS/EOS reserve ids 0/1/2.
    """
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(3, vocab_size, (batch_size, src_len), generator=g)
    tgt = torch.randint(3, vocab_size, (batch_size, tgt_len), generator=g)
    bos = torch.full((batch_size, 1), BOS_ID, dtype=torch.long)
    eos = torch.full((batch_size, 1), EOS_ID, dtype=torch.long)
    tgt_in = torch.cat([bos, tgt], dim=1)         # (B, 1 + L_tgt)
    tgt_out = torch.cat([tgt, eos], dim=1)        # (B, L_tgt + 1)
    if device is not None:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
    return src, tgt_in, tgt_out
