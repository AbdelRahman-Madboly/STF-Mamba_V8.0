#!/usr/bin/env python3
"""
STF-Mamba V8.0 - Hydra-Mamba Bidirectional Temporal Module
============================================================

Adapted from V7.3 hydra_mixer.py — preserved core HydraQuasiseparableMixer,
stripped V7.3-specific components (MamerBlock, PNMamerStage) that depended
on the disproven HLL hypothesis.

Pipeline:
    (B, T, 768) → Linear(768→512) + LayerNorm + GELU → Hydra x2 → (B, T, 512)

Quasiseparable formula:
    M_QS(X) = shift(SS_fwd(X)) + flip(shift(SS_bwd(flip(X)))) + D·X

Why Hydra over Bi-Mamba (V7.3):
    - Single unified matrix → gradients flow through both directions
    - Shared input projections → ~50% fewer params than V7.3's Bi-Mamba
    - Conv1d fallback when mamba_ssm unavailable (Kaggle compatibility)

FALLBACK: When mamba_ssm not installed, uses Conv1d-based sequence mixer.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# =============================================================================
# OPTIONAL MAMBA DEPENDENCY
# =============================================================================

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    logger.info("mamba_ssm available — using optimized Mamba S6")
except ImportError:
    MAMBA_AVAILABLE = False
    logger.warning("mamba_ssm not available — using Conv1d fallback")


# =============================================================================
# CONV1D FALLBACK (from V7.3 hydra_mixer.py — preserved)
# =============================================================================

class Conv1DSequenceMixer(nn.Module):
    """
    Fallback sequence mixer when mamba_ssm is not available.
    Uses causal Conv1d to approximate SSM behavior.
    """

    def __init__(self, d_model: int, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size - 1,  # Causal padding
            groups=d_model,  # Depthwise for efficiency
        )
        self.proj = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)"""
        # Conv1d expects (B, D, T)
        y = x.transpose(1, 2)
        y = self.conv(y)[:, :, :x.shape[1]]  # Trim causal padding
        y = y.transpose(1, 2)
        y = self.act(self.proj(y))
        return y


# =============================================================================
# HYDRA QUASISEPARABLE MIXER (from V7.3 hydra_mixer.py — preserved)
# =============================================================================

class HydraQuasiseparableMixer(nn.Module):
    """
    Quasiseparable Matrix Mixer inspired by Hydra (2024).

    M_QS(X) = shift(SS_fwd(X)) + flip(shift(SS_bwd(flip(X)))) + D·X

    Key insight: Instead of running two independent Mamba blocks and concatenating,
    we run a single forward SSM and a single backward SSM with SHARED input
    projections and ADD their outputs. This creates a unified bidirectional
    representation where gradients flow through both directions from the start.

    Args:
        d_model: Model dimension (input/output).
        d_state: SSM state dimension. Default: 64 (Hydra default).
        d_conv: Local convolution width. Default: 7 (±3 frame context).
        expand: Expansion factor for inner dimension. Default: 2.
        dropout: Dropout rate. Default: 0.1.
        use_shift: Enable shift operation for enhanced local context. Default: True.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 7,
        expand: int = 2,
        dropout: float = 0.1,
        use_shift: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.use_shift = use_shift

        # SHARED input projection (key difference from V7.3 Bi-Mamba!)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        if MAMBA_AVAILABLE:
            self.ssm_fwd = Mamba(
                d_model=self.d_inner,
                d_state=d_state,
                d_conv=d_conv,
                expand=1,  # Already expanded via in_proj
            )
            self.ssm_bwd = Mamba(
                d_model=self.d_inner,
                d_state=d_state,
                d_conv=d_conv,
                expand=1,
            )
        else:
            self.ssm_fwd = Conv1DSequenceMixer(self.d_inner, kernel_size=d_conv)
            self.ssm_bwd = Conv1DSequenceMixer(self.d_inner, kernel_size=d_conv)

        # Diagonal component: D·X (local instant interactions)
        self.diag_weight = nn.Parameter(torch.ones(self.d_inner) * 0.1)

        # Output projection: d_inner → d_model (ADDITIVE, not concat like V7.3)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Learnable gating scale
        self.gate_scale = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quasiseparable forward pass.

        M_QS(X) = shift(SS_fwd(X)) + flip(shift(SS_bwd(flip(X)))) + D·X

        Args:
            x: (B, T, D) input sequence.

        Returns:
            (B, T, D) output sequence with bidirectional context.
        """
        residual = x

        # Shared projection → split into processing and gating
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)  # Each: (B, T, d_inner)

        # Forward SSM
        y_fwd = self.ssm_fwd(x_proj)

        # Apply shift: shift(SS_fwd(X))
        if self.use_shift:
            y_fwd = self._shift_right(y_fwd)

        # Backward SSM: SS_bwd(flip(X))
        x_flipped = torch.flip(x_proj, dims=[1])
        y_bwd = self.ssm_bwd(x_flipped)

        # Apply shift then flip back: flip(shift(SS_bwd(flip(X))))
        if self.use_shift:
            y_bwd = self._shift_right(y_bwd)
        y_bwd = torch.flip(y_bwd, dims=[1])

        # Diagonal component: D·X
        y_diag = x_proj * self.diag_weight.unsqueeze(0).unsqueeze(0)

        # Quasiseparable combination (ADDITIVE, not concat!)
        y = y_fwd + y_bwd + y_diag

        # Gating with SiLU
        y = y * F.silu(z) * self.gate_scale

        # Project back to d_model
        y = self.out_proj(y)
        y = self.dropout(y)

        return self.norm(residual + y)

    def _shift_right(self, x: torch.Tensor) -> torch.Tensor:
        """Shift sequence one step right, padding with zero."""
        pad = torch.zeros_like(x[:, :1, :])
        return torch.cat([pad, x[:, :-1, :]], dim=1)


# =============================================================================
# V8.0 TEMPORAL MODULE (wraps Hydra blocks + projection)
# =============================================================================

class HydraMambaTemporalModule(nn.Module):
    """
    Complete Hydra-Mamba temporal module for V8.0.

    Includes the projection layer (768→512) and N stacked Hydra blocks.

    Pipeline:
        (B, T, 768) → Linear(768→512) → LayerNorm → GELU → Hydra×N → (B, T, 512)

    Args:
        input_dim: Backbone output dimension. Default: 768 (DINOv2-ViT-B/14).
        proj_dim: Temporal projection dimension. Default: 512.
        num_blocks: Number of stacked HydraQuasiseparableMixer blocks. Default: 2.
        d_state: SSM state dimension. Default: 64.
        d_conv: Local convolution width. Default: 7.
        expand: Expansion factor. Default: 2.
        dropout: Dropout rate. Default: 0.1.
        use_shift: Enable shift operation. Default: True.
    """

    def __init__(
        self,
        input_dim: int = 768,
        proj_dim: int = 512,
        num_blocks: int = 2,
        d_state: int = 64,
        d_conv: int = 7,
        expand: int = 2,
        dropout: float = 0.1,
        use_shift: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim

        # Temporal projection: 768 → 512
        self.projection = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        # Stacked Hydra blocks
        self.blocks = nn.ModuleList([
            HydraQuasiseparableMixer(
                d_model=proj_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                use_shift=use_shift,
            )
            for _ in range(num_blocks)
        ])

        logger.info(
            f"HydraMambaTemporalModule: {input_dim}→{proj_dim}, "
            f"{num_blocks} blocks, mamba={'yes' if MAMBA_AVAILABLE else 'Conv1d fallback'}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 768) — per-frame DINOv2 CLS tokens.

        Returns:
            (B, T, 512) — temporally-aware embeddings.
        """
        # Project: (B, T, 768) → (B, T, 512)
        x = self.projection(x)

        # Pass through stacked Hydra blocks
        for block in self.blocks:
            x = block(x)

        return x

    def get_param_groups(self, lr_temporal: float = 1e-4):
        """Returns parameter groups for the optimizer."""
        return [{"params": self.parameters(), "lr": lr_temporal}]


def is_mamba_available() -> bool:
    """Check if mamba_ssm is available."""
    return MAMBA_AVAILABLE
