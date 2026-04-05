"""
Custom learning rate scheduler for the Transformer (Vaswani et al., 2017).

Implements the warmup-then-decay schedule from Section 5.3:
    lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))

This corresponds to linearly increasing the learning rate for the first
`warmup_steps` training steps, then decreasing it proportionally to the
inverse square root of the step number.
"""

import torch
from torch.optim.lr_scheduler import LambdaLR


def transformer_lr_lambda(step: int, d_model: int = 512, warmup_steps: int = 4000) -> float:
    """Compute the LR multiplier for a given step.

    Args:
        step: Current training step (1-indexed internally; 0 is handled).
        d_model: Model dimension.
        warmup_steps: Number of warmup steps.

    Returns:
        Learning rate value (not a multiplier — use with optimizer lr=1.0).
    """
    # Avoid division by zero at step 0
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)


def get_transformer_scheduler(
    optimizer: torch.optim.Optimizer,
    d_model: int = 512,
    warmup_steps: int = 4000,
    last_epoch: int = -1,
) -> LambdaLR:
    """Create the Transformer LR scheduler.

    The optimizer should be initialized with lr=1.0 since the schedule
    computes the absolute learning rate.

    Args:
        optimizer: Optimizer instance (lr should be 1.0).
        d_model: Model dimension.
        warmup_steps: Number of warmup steps.
        last_epoch: The index of last epoch.

    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(step: int) -> float:
        return transformer_lr_lambda(step, d_model, warmup_steps)

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
