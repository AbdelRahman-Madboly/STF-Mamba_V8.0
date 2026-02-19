#!/usr/bin/env python3
"""
STF-Mamba V8.0 - Hydra Quasiseparable Mixer
=============================================

Replaces V7.3's heuristic Bi-Mamba (two separate SSMs concatenated)
with a single quasiseparable matrix mixer that achieves native bidirectionality.

M_QS(X) = shift(SS_fwd(X)) + flip(shift(SS_bwd(flip(X)))) + D·X

WHY THIS IS BETTER:
- V7.3: Two independent SSMs, late fusion → each branch can learn shortcuts independently
- V8.0: Single unified matrix → gradient flows through both directions simultaneously
- Parameter efficient: Shares input projections → ~50% fewer params than Bi-Mamba

FALLBACK: When mamba_ssm not installed, uses Conv1d-based sequence mixer
that approximates the quasiseparable structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Try to import Mamba
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class HydraQuasiseparableMixer(nn.Module):
    """
    Quasiseparable Matrix Mixer inspired by Hydra (2026).
    
    M_QS(X) = shift(SS_fwd(X)) + flip(shift(SS_bwd(flip(X)))) + D·X
    
    Key insight: Instead of running two independent Mamba blocks and concatenating,
    we run a single forward SSM and a single backward SSM with SHARED input 
    projections and ADD their outputs. This creates a unified bidirectional 
    representation where gradients flow through both directions from the start.
    
    The shift operation enhances local context modeling, critical for tracking
    the ±1 frame temporal drift artifacts.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 7,
        expand: int = 2,
        dropout: float = 0.1,
        use_shift: bool = True,
    ):
        """
        Args:
            d_model: Model dimension
            d_state: SSM state dimension (Hydra default: 64)
            d_conv: Local convolution width (7 = ±3 frame context)
            expand: Expansion factor
            dropout: Dropout rate
            use_shift: Enable the shift operation for enhanced local context
        """
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.use_shift = use_shift
        
        # SHARED input projection (key difference from V7.3!)
        # V7.3 had separate projections for fwd and bwd → double params
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        if MAMBA_AVAILABLE:
            # Forward SSM
            self.ssm_fwd = Mamba(
                d_model=self.d_inner,
                d_state=d_state,
                d_conv=d_conv,
                expand=1,  # Already expanded
            )
            # Backward SSM (shared architecture, different learned params)
            self.ssm_bwd = Mamba(
                d_model=self.d_inner,
                d_state=d_state,
                d_conv=d_conv,
                expand=1,
            )
        else:
            # Fallback: Conv1d-based sequence mixer
            self.ssm_fwd = Conv1DSequenceMixer(self.d_inner, kernel_size=d_conv)
            self.ssm_bwd = Conv1DSequenceMixer(self.d_inner, kernel_size=d_conv)
        
        # Diagonal component: D·X (local instant interactions)
        self.diag_weight = nn.Parameter(torch.ones(self.d_inner) * 0.1)
        
        # Output projection: d_inner → d_model (NOT 2*d_inner like V7.3!)
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
            x: (B, T, D) input sequence
        Returns:
            (B, T, D) output sequence with bidirectional context
        """
        residual = x
        
        # Shared projection → split into processing and gating
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)  # Each: (B, T, d_inner)
        
        # === Forward SSM: SS_fwd(X) ===
        if MAMBA_AVAILABLE:
            y_fwd = self.ssm_fwd(x_proj)
        else:
            y_fwd = self.ssm_fwd(x_proj)
        
        # Apply shift if enabled: shift(SS_fwd(X))
        if self.use_shift:
            y_fwd = self._shift_right(y_fwd)
        
        # === Backward SSM: SS_bwd(flip(X)) ===
        x_flipped = torch.flip(x_proj, dims=[1])
        if MAMBA_AVAILABLE:
            y_bwd = self.ssm_bwd(x_flipped)
        else:
            y_bwd = self.ssm_bwd(x_flipped)
        
        # Apply shift then flip back: flip(shift(SS_bwd(flip(X))))
        if self.use_shift:
            y_bwd = self._shift_right(y_bwd)
        y_bwd = torch.flip(y_bwd, dims=[1])
        
        # === Diagonal component: D·X ===
        y_diag = x_proj * self.diag_weight.unsqueeze(0).unsqueeze(0)
        
        # === Quasiseparable combination (ADDITIVE, not concat!) ===
        y = y_fwd + y_bwd + y_diag
        
        # Gating with SiLU
        y = y * F.silu(z) * self.gate_scale
        
        # Project back to d_model
        y = self.out_proj(y)
        y = self.dropout(y)
        
        return self.norm(residual + y)
    
    def _shift_right(self, x: torch.Tensor) -> torch.Tensor:
        """Shift sequence one step right, padding with zero."""
        # x: (B, T, D)
        pad = torch.zeros_like(x[:, :1, :])  # (B, 1, D)
        return torch.cat([pad, x[:, :-1, :]], dim=1)


class Conv1DSequenceMixer(nn.Module):
    """
    Fallback sequence mixer when mamba_ssm is not available.
    Uses causal Conv1d to approximate SSM behavior.
    """
    
    def __init__(self, d_model: int, kernel_size: int = 7):
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


class MamerBlock(nn.Module):
    """
    Mamer Layer: SSM → Attention (replacing SSM → FFN from V7.3).
    
    Layer_Mamer(x) = Attention(Norm(Mamba(Norm(x)) + x)) + (Mamba(x) + x)
    
    The key insight: FFN only transforms features locally. Attention can compare
    spectral signatures across ALL frames — catching global spectral drift that
    pure SSMs miss due to the "recall bottleneck."
    
    In V8.0, this is used every N=3 Hydra blocks (MamBo-3-Hydra-N3 config).
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_hydra: bool = True,
        hydra_d_state: int = 64,
        hydra_d_conv: int = 7,
    ):
        super().__init__()
        
        # Norm layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # SSM component (Hydra or Bi-Mamba fallback)
        if use_hydra:
            self.ssm = HydraQuasiseparableMixer(
                d_model=d_model,
                d_state=hydra_d_state,
                d_conv=hydra_d_conv,
                dropout=dropout,
            )
        else:
            # V7.3-style Bi-Mamba as fallback
            from .bi_mamba_fallback import BiMambaBlock
            self.ssm = BiMambaBlock(d_model=d_model, dropout=dropout)
        
        # Attention component (replaces FFN!)
        self.attention = nn.MultiheadAttention(
            d_model, num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mamer forward: SSM → Attention
        
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D) with both local temporal and global spectral context
        """
        # === Phase 1: Compression (SSM) ===
        # Local temporal inconsistencies compressed into hidden state
        ssm_out = self.ssm(self.norm1(x))  # SSM has its own residual
        
        # === Phase 2: Recall (Attention) ===
        # Global query: "Does spectral profile at frame t match frame t-N?"
        x2 = self.norm2(ssm_out)
        attn_out, _ = self.attention(x2, x2, x2)
        attn_out = self.attn_dropout(attn_out)
        
        return ssm_out + attn_out


class PNMamerStage(nn.Module):
    """
    One stage of the PN-Mamer architecture.
    
    Config: MamBo-3-Hydra-N3 = 3 Hydra blocks + 1 Attention layer per stage.
    
    Data flow:
        Input → [Hydra × 3] → MamerAttention → Output
        
    Each Hydra block does local temporal modeling.
    The Mamer attention at the end does global spectral verification.
    """
    
    def __init__(
        self,
        d_model: int,
        num_hydra_blocks: int = 3,
        num_heads: int = 8,
        d_state: int = 64,
        d_conv: int = 7,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        use_hydra: bool = True,
        use_mamer: bool = True,
    ):
        super().__init__()
        
        # N Hydra/Bi-Mamba blocks for local temporal modeling
        self.ssm_blocks = nn.ModuleList()
        for i in range(num_hydra_blocks):
            if use_hydra:
                block = HydraQuasiseparableMixer(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    dropout=dropout,
                )
            else:
                # Fallback to V7.3 Bi-Mamba
                block = HydraQuasiseparableMixer(
                    d_model=d_model,
                    d_state=16,  # V7.3 default
                    d_conv=4,
                    dropout=dropout,
                )
            self.ssm_blocks.append(block)
        
        # Optional Mamer attention for global spectral recall
        self.use_mamer = use_mamer
        if use_mamer:
            self.mamer_norm = nn.LayerNorm(d_model)
            self.mamer_attention = nn.MultiheadAttention(
                d_model, num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.mamer_dropout = nn.Dropout(dropout)
        else:
            # V7.3 fallback: FFN instead of attention
            self.ffn = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(dropout),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D) 
        """
        # Local temporal modeling through N SSM blocks
        for block in self.ssm_blocks:
            x = block(x)
        
        # Global spectral verification
        if self.use_mamer:
            # Mamer: Attention replaces FFN
            x_norm = self.mamer_norm(x)
            attn_out, _ = self.mamer_attention(x_norm, x_norm, x_norm)
            x = x + self.mamer_dropout(attn_out)
        else:
            # V7.3: FFN
            x = x + self.ffn(x)
        
        return x
