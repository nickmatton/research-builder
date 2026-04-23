"""Training loop.

Implements the paper's optimizer / schedule / loss exactly:
- Adam (β1=0.9, β2=0.98, ε=1e-9)            §5.3, p.7
- LR = d_model^-0.5 · min(step^-0.5, step · warmup^-1.5)   §5.3, eq. 3
- Label smoothing ε_ls = 0.1                §5.4, p.8
- Cross-entropy with PAD ignored

Two modes:
  train.py --overfit-one-batch  → trains on a single fixed synthetic batch
                                  for --max-steps. Loss should collapse to ~0.
  train.py --max-steps N        → trains on fresh synthetic batches per step
                                  (smoke run; loss decreases but not to 0).

Real WMT training is the Phase 4 follow-up — same loop, swap the data source.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from .data import PAD_ID, synthetic_batch
from .transformer import Transformer


def lr_at_step(step: int, d_model: int, warmup: int) -> float:
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy with label smoothing (Szegedy 2016, paper §5.4).

    Distribute (1 - ε) probability mass on the true class and ε / (V-1) on
    the rest. PAD positions contribute 0 to loss.
    """

    def __init__(self, vocab_size: int, smoothing: float = 0.1, pad_id: int = 0) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.pad_id = pad_id

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, L, V); target: (B, L)
        log_probs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            true_dist.scatter_(2, target.unsqueeze(2), 1.0 - self.smoothing)
            true_dist[:, :, self.pad_id] = 0
            mask = (target != self.pad_id).unsqueeze(2).float()
            true_dist = true_dist * mask
        # KL divergence; sum over vocab, mean over non-pad positions
        loss = -(true_dist * log_probs).sum(dim=-1)
        denom = (target != self.pad_id).float().sum().clamp(min=1)
        return loss.sum() / denom


def train(
    overfit_one_batch: bool,
    max_steps: int,
    d_model: int,
    num_heads: int,
    num_layers: int,
    d_ff: int,
    vocab_size: int,
    batch_size: int,
    src_len: int,
    tgt_len: int,
    warmup: int,
    output_dir: Path,
    label_smoothing: float = 0.1,
    log_every: int = 50,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_ff,
        dropout=0.0 if overfit_one_batch else 0.1,
        pad_id=PAD_ID,
    )
    print(f"[train] model has {model.num_parameters():,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    loss_fn = LabelSmoothingLoss(vocab_size=vocab_size, smoothing=label_smoothing, pad_id=PAD_ID)

    fixed_batch = None
    if overfit_one_batch:
        fixed_batch = synthetic_batch(batch_size, src_len, tgt_len, vocab_size, seed=0)

    history: list[tuple[int, float, float]] = []
    start = time.time()
    final_loss = float("inf")
    for step in range(1, max_steps + 1):
        if overfit_one_batch:
            src, tgt_in, tgt_out = fixed_batch
        else:
            src, tgt_in, tgt_out = synthetic_batch(batch_size, src_len, tgt_len, vocab_size, seed=step)

        lr = lr_at_step(step, d_model, warmup)
        for g in optimizer.param_groups:
            g["lr"] = lr

        logits = model(src, tgt_in)                     # (B, L_tgt+1, V)
        # Predict tgt_out from tgt_in. Both are length tgt_len+1 (BOS + tgt; tgt + EOS).
        loss = loss_fn(logits, tgt_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == max_steps:
            print(f"[step {step:5d}] loss={loss.item():.4f} lr={lr:.2e}")
            history.append((step, loss.item(), lr))
        final_loss = loss.item()

    duration = time.time() - start
    print(f"[train] {max_steps} steps in {duration:.1f}s. Final loss: {final_loss:.4f}")

    metrics = {
        "max_steps": max_steps,
        "final_loss": final_loss,
        "wall_clock_seconds": duration,
        "param_count": model.num_parameters(),
        "overfit_one_batch": overfit_one_batch,
        "history": [{"step": s, "loss": l, "lr": lr} for s, l, lr in history],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--overfit-one-batch", action="store_true")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--src-len", type=int, default=6)
    p.add_argument("--tgt-len", type=int, default=6)
    p.add_argument("--warmup", type=int, default=400)
    p.add_argument("--label-smoothing", type=float, default=0.1,
                   help="Set to 0 for sanity overfit (paper uses 0.1)")
    p.add_argument("--output-dir", type=Path, default=Path("runs/dev"))
    args = p.parse_args()

    train(
        overfit_one_batch=args.overfit_one_batch,
        max_steps=args.max_steps,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        src_len=args.src_len,
        tgt_len=args.tgt_len,
        warmup=args.warmup,
        label_smoothing=args.label_smoothing,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
