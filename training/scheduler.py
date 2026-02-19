#!/usr/bin/env python3
"""
STF-Mamba V8.0 - Learning Rate Scheduler
==========================================

Linear warmup for first N epochs → cosine annealing to 0.

Kaggle:  warmup_epochs=3,  total_epochs=25
RunPod:  warmup_epochs=5,  total_epochs=50
"""

import logging
import math

import torch
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int = 3,
) -> LambdaLR:
    """
    Build warmup + cosine annealing scheduler.

    Args:
        optimizer: Configured optimizer.
        total_epochs: Total training epochs.
        warmup_epochs: Linear warmup epochs.

    Returns:
        LambdaLR scheduler (call .step() once per epoch).
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            # Linear warmup: 0 → 1
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing: 1 → 0
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    logger.info(
        f"Scheduler: {warmup_epochs}-epoch warmup → "
        f"cosine annealing over {total_epochs} epochs"
    )
    return scheduler
