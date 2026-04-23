"""
Training script for the Transformer model (Vaswani et al., 2017).

Implements the full training loop with:
- Adam optimizer (β₁=0.9, β₂=0.98, ε=10⁻⁹)
- Custom warmup-then-decay LR schedule
- Label smoothing (ε_ls=0.1)
- Checkpoint saving at regular intervals
- Training logging to JSON

Usage:
    python train.py [--config base|big] [--max_steps N] [--data_path PATH]
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup — make architecture & data outputs importable
# ---------------------------------------------------------------------------
_PHASE_ROOT = Path(__file__).resolve().parent.parent  # phases/training/1
_ARCH_DIR = _PHASE_ROOT.parent.parent / "architecture" / "1" / "outputs"
_DATA_DIR = _PHASE_ROOT.parent.parent / "data" / "1" / "outputs"

sys.path.insert(0, str(_ARCH_DIR))
sys.path.insert(0, str(_DATA_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for lr_scheduler

from config import TransformerConfig
from transformer import Transformer
from dataloader import TranslationDataset, create_dataloader, PAD_ID
from lr_scheduler import get_transformer_scheduler


# ---------------------------------------------------------------------------
# Label-smoothing cross-entropy loss
# ---------------------------------------------------------------------------
class LabelSmoothingLoss(nn.Module):
    """KL-divergence loss with label smoothing.

    Distributes ε_ls probability mass uniformly over the vocabulary and
    assigns (1 - ε_ls) to the correct label.  Ignores padding tokens.

    This is equivalent to KL(q || p) where q is the smoothed target
    distribution and p is the model's predicted distribution.
    """

    def __init__(self, vocab_size: int, padding_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, V) raw scores from the model
            target: (N,) ground-truth token indices

        Returns:
            Scalar loss (mean over non-pad tokens).
        """
        log_probs = F.log_softmax(logits, dim=-1)  # (N, V)

        # Smooth target distribution: uniform ε_ls / V, plus (1-ε_ls) on gold
        smooth_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
        smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        smooth_dist[:, self.padding_idx] = 0  # never predict padding

        # Zero out positions where target is pad
        pad_mask = target == self.padding_idx
        smooth_dist[pad_mask] = 0

        loss = -(smooth_dist * log_probs).sum(dim=-1)  # (N,)
        # Mean over non-pad tokens
        non_pad = (~pad_mask).sum().clamp(min=1)
        return loss.sum() / non_pad


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------
def make_src_mask(src: torch.Tensor, pad_idx: int = PAD_ID) -> torch.Tensor:
    """(B, 1, 1, S) – True where token is NOT pad."""
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = PAD_ID) -> torch.Tensor:
    """Combine causal mask and padding mask for the decoder.

    Returns (B, 1, T, T) boolean mask.
    """
    B, T = tgt.shape
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
    causal_mask = torch.tril(torch.ones(T, T, device=tgt.device, dtype=torch.bool))  # (T, T)
    return pad_mask & causal_mask.unsqueeze(0).unsqueeze(0)  # (B, 1, T, T)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(
    config_name: str = "base",
    data_path: str | None = None,
    max_steps: int = 100_000,
    tokens_per_batch: int = 25_000,
    checkpoint_interval_minutes: float = 10.0,
    checkpoint_dir: str | None = None,
    log_path: str | None = None,
    device: str | None = None,
    warmup_steps: int = 4000,
    label_smoothing: float = 0.1,
    log_interval: int = 100,
    vocab_size: int | None = None,
):
    """Run the training loop.

    Returns the final training log list (also saved to disk).
    """
    # ---- Paths ----------------------------------------------------------
    outputs_dir = Path(__file__).resolve().parent
    if data_path is None:
        data_path = str(_DATA_DIR / "en_de_tokenized.pt")
    if checkpoint_dir is None:
        checkpoint_dir = str(outputs_dir / "checkpoints")
    if log_path is None:
        log_path = str(outputs_dir / "training_log.json")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ---- Device ---------------------------------------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Training on {device}")

    # ---- Dataset --------------------------------------------------------
    dataset = TranslationDataset(data_path)
    dataloader = create_dataloader(dataset, tokens_per_batch=tokens_per_batch, shuffle=True)
    actual_vocab = vocab_size or dataset.metadata.get("vocab_size", 1373)

    # ---- Model ----------------------------------------------------------
    if config_name == "big":
        cfg = TransformerConfig.big(src_vocab_size=actual_vocab, tgt_vocab_size=actual_vocab)
    else:
        cfg = TransformerConfig.base(src_vocab_size=actual_vocab, tgt_vocab_size=actual_vocab)

    model = Transformer(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # ---- Optimizer & scheduler ------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = get_transformer_scheduler(
        optimizer, d_model=cfg.d_model, warmup_steps=warmup_steps
    )

    # ---- Loss -----------------------------------------------------------
    criterion = LabelSmoothingLoss(
        vocab_size=actual_vocab, padding_idx=PAD_ID, smoothing=label_smoothing
    )

    # ---- Training loop --------------------------------------------------
    model.train()
    global_step = 0
    training_log = []
    last_ckpt_time = time.time()
    epoch = 0

    while global_step < max_steps:
        epoch += 1
        for batch in dataloader:
            if global_step >= max_steps:
                break

            src = batch["src"].to(device)      # (B, S)
            tgt = batch["tgt"].to(device)      # (B, T)

            # Decoder input: all tokens except last; target: all tokens except first
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = make_src_mask(src)
            tgt_mask = make_tgt_mask(tgt_input)

            # Forward
            logits = model(src, tgt_input, src_mask, tgt_mask)  # (B, T-1, V)

            # Reshape for loss
            logits_flat = logits.reshape(-1, logits.size(-1))
            tgt_flat = tgt_output.reshape(-1)

            loss = criterion(logits_flat, tgt_flat)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (not in paper but helps stability)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            global_step += 1
            current_lr = scheduler.get_last_lr()[0]

            # Check for NaN/Inf
            if not math.isfinite(loss.item()):
                print(f"WARNING: Non-finite loss at step {global_step}: {loss.item()}")

            # Logging
            if global_step % log_interval == 0 or global_step == 1:
                log_entry = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": loss.item(),
                    "lr": current_lr,
                    "time": time.time(),
                }
                training_log.append(log_entry)
                print(
                    f"Step {global_step:>6d} | Loss {loss.item():.4f} | "
                    f"LR {current_lr:.2e} | Epoch {epoch}"
                )

            # Checkpoint by time interval
            now = time.time()
            if (now - last_ckpt_time) >= checkpoint_interval_minutes * 60:
                ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_step{global_step}.pt")
                save_checkpoint(model, optimizer, scheduler, global_step, epoch, ckpt_path)
                last_ckpt_time = now
                print(f"  -> Saved checkpoint: {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(checkpoint_dir, f"checkpoint_step{global_step}.pt")
    save_checkpoint(model, optimizer, scheduler, global_step, epoch, final_path)
    print(f"  -> Saved final checkpoint: {final_path}")

    # Save training log
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"Training log saved to {log_path}")

    return training_log


def save_checkpoint(model, optimizer, scheduler, step, epoch, path):
    """Save a training checkpoint."""
    ckpt = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "config": model.config,
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load a training checkpoint.

    Returns (step, epoch).
    """
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["step"], ckpt["epoch"]


def average_checkpoints(checkpoint_paths: list[str], model: nn.Module) -> nn.Module:
    """Average model weights across multiple checkpoints.

    Used for the final model: average last 5 (base) or last 20 (big) checkpoints.

    Args:
        checkpoint_paths: Paths to checkpoint files.
        model: Model instance (weights will be overwritten).

    Returns:
        Model with averaged weights.
    """
    avg_state = None
    n = len(checkpoint_paths)

    for path in checkpoint_paths:
        ckpt = torch.load(path, weights_only=False)
        state = ckpt["model_state_dict"]
        if avg_state is None:
            avg_state = {k: v.float().clone() for k, v in state.items()}
        else:
            for k in avg_state:
                avg_state[k] += state[k].float()

    for k in avg_state:
        avg_state[k] /= n

    model.load_state_dict(avg_state)
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Transformer")
    parser.add_argument("--config", default="base", choices=["base", "big"])
    parser.add_argument("--max_steps", type=int, default=100_000)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--tokens_per_batch", type=int, default=25_000)
    parser.add_argument("--checkpoint_interval_minutes", type=float, default=10.0)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--log_interval", type=int, default=100)
    args = parser.parse_args()

    train(
        config_name=args.config,
        data_path=args.data_path,
        max_steps=args.max_steps,
        tokens_per_batch=args.tokens_per_batch,
        checkpoint_interval_minutes=args.checkpoint_interval_minutes,
        checkpoint_dir=args.checkpoint_dir,
        log_path=args.log_path,
        device=args.device,
        warmup_steps=args.warmup_steps,
        label_smoothing=args.label_smoothing,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
