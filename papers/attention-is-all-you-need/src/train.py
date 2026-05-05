"""Training loop.

Implements the paper's optimizer / schedule / loss exactly:
- Adam (β1=0.9, β2=0.98, ε=1e-9)            §5.3, p.7
- LR = d_model^-0.5 · min(step^-0.5, step · warmup^-1.5)   §5.3, eq. 3
- Label smoothing ε_ls = 0.1                §5.4, p.8
- Cross-entropy with PAD ignored

Three modes:
  --overfit-one-batch              fixed synthetic batch; loss → 0 (sanity)
  --data synthetic                 fresh synthetic per step (smoke)
  --data wmt --tokenizer ...       real WMT 2014 EN-DE via src.wmt + src.tokenize
                                   (token-budget batches, paper §5.1)

Real WMT mode also writes a checkpoint at the end (model + optimizer state).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from . import tokenize as tk
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
        log_probs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            true_dist.scatter_(2, target.unsqueeze(2), 1.0 - self.smoothing)
            true_dist[:, :, self.pad_id] = 0
            mask = (target != self.pad_id).unsqueeze(2).float()
            true_dist = true_dist * mask
        loss = -(true_dist * log_probs).sum(dim=-1)
        denom = (target != self.pad_id).float().sum().clamp(min=1)
        return loss.sum() / denom


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _wmt_batches(tokenizer_path: Path, split: str, limit: int | None,
                  token_budget: int, max_len: int):
    """Lazy WMT iterator. Loads + tokenizes once, then yields batches forever."""
    from . import wmt
    print(f"[train] loading WMT 2014 EN-DE ({split}, limit={limit})...")
    pairs = wmt.load_wmt14_en_de(split=split, limit=limit)
    print(f"[train] loaded {len(pairs)} pairs; loading tokenizer from {tokenizer_path}")
    tok = tk.load(tokenizer_path)
    print(f"[train] tokenizing pairs (max_len={max_len})...")
    ids = wmt.tokenize_pairs(pairs, tok, direction="en-de", max_len=max_len)
    print(f"[train] {len(ids)} pairs after length filter; building token-budget batches "
          f"(budget={token_budget})")
    epoch = 0
    while True:
        epoch += 1
        for batch in wmt.token_budget_batches(ids, token_budget=token_budget, seed=epoch):
            yield batch.src, batch.tgt_in, batch.tgt_out


def train(
    *,
    overfit_one_batch: bool,
    data_mode: str,                # "synthetic" | "wmt"
    max_steps: int,
    d_model: int,
    num_heads: int,
    num_layers: int,
    d_ff: int,
    vocab_size: int,               # used only for synthetic; wmt overrides via tokenizer
    batch_size: int,
    src_len: int,
    tgt_len: int,
    warmup: int,
    output_dir: Path,
    label_smoothing: float = 0.1,
    log_every: int = 50,
    save_checkpoint: bool = False,
    tokenizer_path: Path | None = None,
    wmt_split: str = "train",
    wmt_limit: int | None = None,
    token_budget: int = 25000,
    max_len: int = 256,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device()
    print(f"[train] device={device}")

    # Resolve vocab size from tokenizer when in WMT mode.
    if data_mode == "wmt":
        if tokenizer_path is None or not tokenizer_path.exists():
            raise FileNotFoundError(
                f"--tokenizer required and must exist when --data wmt. Got: {tokenizer_path}"
            )
        vocab_size = tk.vocab_size(tk.load(tokenizer_path))
        print(f"[train] WMT mode; vocab_size from tokenizer: {vocab_size}")

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
    ).to(device)
    print(f"[train] model has {model.num_parameters():,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    loss_fn = LabelSmoothingLoss(vocab_size=vocab_size, smoothing=label_smoothing,
                                  pad_id=PAD_ID).to(device)

    # Pick a batch source.
    fixed_batch = None
    wmt_iter = None
    if overfit_one_batch:
        s, ti, to = synthetic_batch(batch_size, src_len, tgt_len, vocab_size, seed=0)
        fixed_batch = (s.to(device), ti.to(device), to.to(device))
    elif data_mode == "wmt":
        wmt_iter = _wmt_batches(tokenizer_path, wmt_split, wmt_limit, token_budget, max_len)

    history: list[tuple[int, float, float]] = []
    start = time.time()
    final_loss = float("inf")
    for step in range(1, max_steps + 1):
        if overfit_one_batch:
            src, tgt_in, tgt_out = fixed_batch
        elif data_mode == "wmt":
            src, tgt_in, tgt_out = next(wmt_iter)
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        else:
            src, tgt_in, tgt_out = synthetic_batch(batch_size, src_len, tgt_len, vocab_size, seed=step)
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

        lr = lr_at_step(step, d_model, warmup)
        for g in optimizer.param_groups:
            g["lr"] = lr

        logits = model(src, tgt_in)
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

    if save_checkpoint:
        ckpt_path = output_dir / "checkpoint.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": max_steps,
            "config": {
                "d_model": d_model, "num_heads": num_heads, "num_layers": num_layers,
                "d_ff": d_ff, "vocab_size": vocab_size, "warmup": warmup,
            },
        }, ckpt_path)
        print(f"[train] checkpoint saved → {ckpt_path}")

    metrics = {
        "max_steps": max_steps,
        "final_loss": final_loss,
        "wall_clock_seconds": duration,
        "param_count": model.num_parameters(),
        "overfit_one_batch": overfit_one_batch,
        "data_mode": data_mode,
        "history": [{"step": s, "loss": l, "lr": lr} for s, l, lr in history],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--overfit-one-batch", action="store_true")
    p.add_argument("--data", default="synthetic", choices=["synthetic", "wmt"],
                   help="synthetic = random ids; wmt = real WMT 2014 EN-DE via HF datasets")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=100,
                   help="synthetic mode only; wmt mode reads from tokenizer")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--src-len", type=int, default=6)
    p.add_argument("--tgt-len", type=int, default=6)
    p.add_argument("--warmup", type=int, default=400)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--output-dir", type=Path, default=Path("runs/dev"))
    p.add_argument("--save-checkpoint", action="store_true")
    p.add_argument("--tokenizer", type=Path, default=None,
                   help="path to tokenizer.json (required for --data wmt)")
    p.add_argument("--wmt-split", default="train")
    p.add_argument("--wmt-limit", type=int, default=None,
                   help="cap WMT pairs (smoke runs); None = full split")
    p.add_argument("--token-budget", type=int, default=25000,
                   help="paper §5.1: ~25000 source + 25000 target tokens per batch")
    p.add_argument("--max-len", type=int, default=256,
                   help="drop pairs whose either side exceeds this token count")
    args = p.parse_args()

    train(
        overfit_one_batch=args.overfit_one_batch,
        data_mode=args.data,
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
        save_checkpoint=args.save_checkpoint,
        tokenizer_path=args.tokenizer,
        wmt_split=args.wmt_split,
        wmt_limit=args.wmt_limit,
        token_budget=args.token_budget,
        max_len=args.max_len,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
