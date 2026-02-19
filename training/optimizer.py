#!/usr/bin/env python3
"""
STF-Mamba V8.0 - Optimizer Configuration
==========================================

Differential learning rates:
    - DINOv2 fine-tune blocks (10-11): lr=5e-6
    - Hydra-Mamba temporal:             lr=1e-4
    - Consistency head:                 lr=1e-4

AdamW with weight_decay=1e-4, gradient clipping at 1.0.
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def build_optimizer(
    model: nn.Module,
    lr_backbone: float = 5e-6,
    lr_temporal: float = 1e-4,
    lr_head: float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """
    Build AdamW optimizer with differential learning rates.

    Args:
        model: STFMambaV8 model instance.
        lr_backbone: LR for DINOv2 fine-tuned blocks.
        lr_temporal: LR for Hydra-Mamba + projection.
        lr_head: LR for consistency head.
        weight_decay: L2 regularization weight.

    Returns:
        Configured AdamW optimizer.
    """
    param_groups = model.get_param_groups(
        lr_backbone=lr_backbone,
        lr_temporal=lr_temporal,
        lr_head=lr_head,
    )

    # Add weight decay to all groups
    for group in param_groups:
        group["weight_decay"] = weight_decay

    optimizer = torch.optim.AdamW(param_groups)

    # Log configuration
    total_params = sum(p.numel() for g in param_groups for p in g["params"])
    for i, g in enumerate(param_groups):
        n_params = sum(p.numel() for p in g["params"])
        logger.info(
            f"  Optimizer group {i}: lr={g['lr']:.1e}, "
            f"params={n_params/1e6:.1f}M, wd={g['weight_decay']}"
        )
    logger.info(f"  Total optimized: {total_params/1e6:.1f}M params")

    return optimizer


def clip_gradients(
    model: nn.Module,
    max_norm: float = 1.0,
) -> float:
    """
    Clip gradients by global norm.

    Args:
        model: Model with computed gradients.
        max_norm: Maximum gradient norm.

    Returns:
        Total gradient norm before clipping.
    """
    return torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad and p.grad is not None],
        max_norm=max_norm,
    ).item()
