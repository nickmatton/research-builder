"""
Checkpoint averaging utility for the Transformer model.

As described in Section 6.1 of 'Attention Is All You Need':
- Base model: average last 5 checkpoints (written at 10-minute intervals)
- Big model: average last 20 checkpoints

Usage:
    python average_checkpoints.py --checkpoint_dir PATH --num_checkpoints 5 --output PATH
"""

import argparse
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Make architecture modules importable
_PHASE_ROOT = Path(__file__).resolve().parent.parent  # phases/eval/1
_ARCH_DIR = _PHASE_ROOT.parent.parent / "architecture" / "1" / "outputs"
_DATA_DIR = _PHASE_ROOT.parent.parent / "data" / "1" / "outputs"
sys.path.insert(0, str(_ARCH_DIR))
sys.path.insert(0, str(_DATA_DIR))

from config import TransformerConfig
from transformer import Transformer


def get_checkpoint_paths(checkpoint_dir: str, num_checkpoints: int = 5) -> list[str]:
    """Get paths to the last N checkpoints sorted by step number.

    Args:
        checkpoint_dir: Directory containing checkpoint files.
        num_checkpoints: Number of most recent checkpoints to use.

    Returns:
        List of checkpoint file paths, sorted by step (ascending).
    """
    pattern = re.compile(r"checkpoint_step(\d+)\.pt$")
    checkpoints = []

    for fname in os.listdir(checkpoint_dir):
        match = pattern.match(fname)
        if match:
            step = int(match.group(1))
            checkpoints.append((step, os.path.join(checkpoint_dir, fname)))

    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])

    # Take last N
    selected = checkpoints[-num_checkpoints:]
    return [path for _, path in selected]


def average_checkpoints(
    checkpoint_paths: list[str],
    device: str = "cpu",
) -> tuple[dict, TransformerConfig]:
    """Average model weights across multiple checkpoints.

    Args:
        checkpoint_paths: Paths to checkpoint files.
        device: Device to load checkpoints onto.

    Returns:
        Tuple of (averaged_state_dict, config).
    """
    avg_state = None
    config = None
    n = len(checkpoint_paths)

    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state = ckpt["model_state_dict"]

        if config is None:
            config = ckpt["config"]

        if avg_state is None:
            avg_state = {k: v.float().clone() for k, v in state.items()}
        else:
            for k in avg_state:
                avg_state[k] += state[k].float()

    for k in avg_state:
        avg_state[k] /= n

    return avg_state, config


def load_averaged_model(
    checkpoint_dir: str,
    num_checkpoints: int = 5,
    device: str = "cpu",
) -> Transformer:
    """Load a model with averaged checkpoint weights.

    Args:
        checkpoint_dir: Directory containing checkpoint files.
        num_checkpoints: Number of most recent checkpoints to average.
        device: Device to load model onto.

    Returns:
        Transformer model with averaged weights.
    """
    paths = get_checkpoint_paths(checkpoint_dir, num_checkpoints)
    if not paths:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")

    print(f"Averaging {len(paths)} checkpoints:")
    for p in paths:
        print(f"  {p}")

    avg_state, config = average_checkpoints(paths, device=device)

    model = Transformer(config).to(device)
    model.load_state_dict(avg_state)
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser(description="Average Transformer checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing checkpoints")
    parser.add_argument("--num_checkpoints", type=int, default=5,
                        help="Number of last checkpoints to average (default: 5)")
    parser.add_argument("--output", type=str, default="averaged_model.pt",
                        help="Output path for averaged checkpoint")
    args = parser.parse_args()

    paths = get_checkpoint_paths(args.checkpoint_dir, args.num_checkpoints)
    if not paths:
        print("No checkpoints found!")
        return

    avg_state, config = average_checkpoints(paths)

    torch.save({
        "model_state_dict": avg_state,
        "config": config,
        "source_checkpoints": paths,
        "num_averaged": len(paths),
    }, args.output)

    print(f"Averaged {len(paths)} checkpoints -> {args.output}")


if __name__ == "__main__":
    main()
