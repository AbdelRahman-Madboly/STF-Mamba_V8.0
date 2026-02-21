#!/usr/bin/env python3
"""
STF-Mamba V8.0 - Loss Functions
=================================

L_total = L_CE + λ * L_var

L_CE:  CrossEntropyLoss(label_smoothing=0.0)
       NEVER use label_smoothing > 0 for binary classification.
       Bug #1 proof: with K=2, smoothed target [0.95, 0.05] means
       perfect prediction loss (1.0) > random loss (0.693).

L_var: Variance auxiliary loss — encourages the model to use the
       temporal consistency signal explicitly:
       L_var = mean(σ²|real) - mean(σ²|fake)
       Minimizing this maximizes the variance gap between classes.
       (We want low variance for real, high variance for fake.)

λ:     0.1 (from config)
"""

import torch
import torch.nn as nn


class STFMambaLoss(nn.Module):
    """
    Combined loss for STF-Mamba V8.0.

    L_total = L_CE + lambda_var * L_var

    Args:
        lambda_var: Weight for variance auxiliary loss. Default: 0.1.
        label_smoothing: Must be 0.0. Exists only as a safeguard
            assertion — will raise error if set > 0.
    """

    def __init__(
        self,
        lambda_var: float = 0.1,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()

        # SAFEGUARD: Never allow label smoothing for binary classification
        if label_smoothing > 0.0:
            raise ValueError(
                f"label_smoothing={label_smoothing} is FORBIDDEN for binary "
                f"classification (K=2). Bug #1: smoothed target [0.95, 0.05] "
                f"makes perfect prediction loss > random loss. "
                f"Always use label_smoothing=0.0."
            )

        self.lambda_var = lambda_var
        self.margin = 1.5  # Fixed margin for log-variance separation
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.0)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        variance: torch.Tensor,
    ):
        """
        Args:
            logits: (B, 2) — classification logits from consistency head.
            labels: (B,) — ground truth labels (0=real, 1=fake).
            variance: (B, 1) — temporal identity variance per clip.

        Returns:
            dict with:
                'total': scalar — total loss for backward().
                'ce': scalar — cross-entropy component.
                'var': scalar — variance auxiliary component.
                'var_gap': scalar — mean(σ²|fake) - mean(σ²|real), for logging.
        """
        # Cross-entropy loss
        loss_ce = self.ce_loss(logits, labels)

        # Variance auxiliary loss
        variance_flat = variance.squeeze(-1)  # (B,)
        real_mask = (labels == 0)
        fake_mask = (labels == 1)

        # Compute mean log-variance per class
        loss_var = torch.tensor(0.0, device=logits.device, requires_grad=True)
        var_gap = torch.tensor(0.0, device=logits.device)

        if real_mask.any() and fake_mask.any():
            v_real = variance_flat[real_mask].mean()
            v_fake = variance_flat[fake_mask].mean()
            
            # MARGIN LOSS: Only penalize if (v_fake - v_real) < margin
            # This prevents loss explosion and encourages stable separation
            loss_var = torch.clamp(self.margin - (v_fake - v_real), min=0.0)
            var_gap = (v_fake - v_real).detach()

        # Total loss
        loss_total = loss_ce + self.lambda_var * loss_var

        return {
            "total": loss_total,
            "ce": loss_ce.detach(),
            "var": loss_var,  # REMOVED .detach() to ensure grad flow
            "var_gap": var_gap,
        }
