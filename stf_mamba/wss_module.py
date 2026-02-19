#!/usr/bin/env python3
"""
STF-Mamba V8.0 - Wavelet-Selective Scan (W-SS) Module
======================================================

V7.3 PROBLEM: DWT sub-bands were flattened into spatial patches, losing 
their frequency-domain meaning. HLL (temporal flicker) was treated the 
same as HHH (spatial noise).

V8.0 FIX: Two specialized scanning streams:
  Stream A (Temporal-Flicker): 1D Hydra scan along temporal axis of HLL
  Stream B (Spatial-Texture): Channel-selective scan of HHH/HLH with 
                               dynamic band priority based on compression

Fusion: F_fused = F_spatial ⊙ σ(F_temporal)
  → Spatial noise only matters if there's concurrent temporal flicker
  → Massive false positive reduction from complex backgrounds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class TemporalFlickerStream(nn.Module):
    """
    Stream A: Process HLL sub-band purely along temporal axis.
    
    HLL = High-Temporal, Low-Height, Low-Width → pure temporal flicker.
    Since HLL inherently represents time, spatial scanning is REDUNDANT.
    
    Pipeline:
      S_HLL (B, C, T/2, H/2, W/2) 
        → Spatial pool → (B, C, T/2) 
        → 1D sequence model → (B, T/2, d_out)
    """
    
    def __init__(self, in_channels: int, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        
        # Spatial pooling: collapse H, W dimensions
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # (B, C, T/2, 1, 1)
        
        # Project to output dim
        self.proj = nn.Linear(in_channels, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        
        # 1D temporal sequence model (lightweight)
        # Using Conv1d-GRU combo for efficiency on small budgets
        self.temporal_conv = nn.Conv1d(
            out_dim, out_dim, kernel_size=5, padding=2, groups=out_dim
        )
        self.temporal_gru = nn.GRU(
            out_dim, out_dim // 2, 
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.out_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hll: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hll: HLL sub-band (B, C, T/2, H/2, W/2)
        Returns:
            F_temporal: (B, T/2, out_dim) — temporal flicker trajectory
        """
        B, C, T, H, W = hll.shape
        
        # Spatial pool: (B, C, T, H, W) → (B, C, T, 1, 1) → (B, C, T)
        x = self.spatial_pool(hll).squeeze(-1).squeeze(-1)  # (B, C, T)
        
        # Permute to (B, T, C) for linear projection
        x = x.permute(0, 2, 1)  # (B, T, C)
        x = self.proj(x)         # (B, T, out_dim)
        x = self.norm(x)
        
        # Temporal conv: (B, out_dim, T) 
        x_conv = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.dropout(x_conv)
        
        # Bidirectional GRU for temporal context
        x, _ = self.temporal_gru(x)  # (B, T, out_dim)
        x = self.out_norm(x)
        
        return x


class SpatialTextureStream(nn.Module):
    """
    Stream B: Process high-frequency spatial sub-bands (HHH, HLH).
    
    Uses channel-selective scanning: dynamically selects which frequency 
    channels to prioritize based on the video's compression profile.
    
    Key insight: Compression attacks different sub-bands differently.
    H.264 at QP=40 destroys HHH but preserves HLH. The scanner learns
    to down-weight destroyed bands per-input.
    
    Pipeline:
      [S_HHH, S_HLH] (B, 2C, T/2, H/2, W/2)
        → Patchify → (B, T/2, num_patches, 2C)
        → Channel attention → weighted features
        → Spatial pool → (B, T/2, d_out)
    """
    
    def __init__(self, in_channels: int, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        
        # Channel attention for dynamic band selection
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # Global pool → (B, 2C, 1, 1, 1)
            nn.Flatten(1),
            nn.Linear(in_channels * 2, in_channels * 2 // 4),
            nn.ReLU(),
            nn.Linear(in_channels * 2 // 4, in_channels * 2),
            nn.Sigmoid(),
        )
        
        # Spatial processing
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(in_channels * 2, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_dim),
            nn.GELU(),
        )
        
        # Collapse spatial dims
        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.out_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hhh: torch.Tensor, hlh: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hhh: HHH sub-band (B, C, T/2, H/2, W/2)
            hlh: HLH sub-band (B, C, T/2, H/2, W/2)
        Returns:
            F_spatial: (B, T/2, out_dim) — spatial texture features
        """
        # Concatenate along channel dim
        x = torch.cat([hhh, hlh], dim=1)  # (B, 2C, T/2, H/2, W/2)
        
        # Channel-selective attention (per-input band weighting)
        gate = self.channel_gate(x)  # (B, 2C)
        gate = gate.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, 2C, 1, 1, 1)
        x = x * gate  # Weight bands by compression profile
        
        # Spatial convolution
        x = self.spatial_conv(x)  # (B, out_dim, T/2, H/2, W/2)
        
        # Collapse spatial → (B, out_dim, T/2)
        x = self.pool(x).squeeze(-1).squeeze(-1)  # (B, out_dim, T/2)
        x = x.permute(0, 2, 1)  # (B, T/2, out_dim)
        x = self.out_norm(x)
        
        return x


class WaveletSelectiveScan(nn.Module):
    """
    W-SS Module: Combines temporal flicker and spatial texture streams.
    
    Fusion: F_fused = F_spatial ⊙ σ(F_temporal)
    
    The gating means: "Only attend to spatial noise IF temporal flicker 
    is also present" → eliminates false positives from natural textures
    (leaves, water, hair) that have high spatial frequency but smooth 
    temporal consistency.
    
    When use_wss=False (Phase A), this falls back to simple HLL 
    concatenation like V7.3.
    """
    
    def __init__(
        self, 
        in_channels: int,
        out_dim: int = 256,
        temporal_dim: int = 128,
        spatial_dim: int = 128,
        dropout: float = 0.1,
        use_wss: bool = True,
    ):
        super().__init__()
        self.use_wss = use_wss
        self.out_dim = out_dim
        
        if use_wss:
            self.temporal_stream = TemporalFlickerStream(
                in_channels, temporal_dim, dropout
            )
            self.spatial_stream = SpatialTextureStream(
                in_channels, spatial_dim, dropout
            )
            # Fusion projection
            self.fusion_proj = nn.Linear(spatial_dim, out_dim)
            self.temporal_gate_proj = nn.Linear(temporal_dim, spatial_dim)
        else:
            # V7.3 fallback: simple HLL attention extraction
            self.hll_proj = nn.Sequential(
                nn.AdaptiveAvgPool3d((None, 1, 1)),
            )
            self.fallback_proj = nn.Linear(in_channels, out_dim)
    
    def forward(
        self, 
        subbands: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            subbands: Dict of DWT sub-bands, each (B, C, T/2, H/2, W/2)
            
        Returns:
            features: (B, T/2, out_dim) — fused frequency features for temporal modeling
            hll_energy: (B,) — HLL energy for contrastive loss
        """
        hll = subbands['HLL']
        B = hll.shape[0]
        
        # Always compute HLL energy for loss
        hll_energy = hll.reshape(B, -1).norm(dim=1)
        
        if self.use_wss:
            # Stream A: Temporal flicker from HLL
            f_temporal = self.temporal_stream(hll)  # (B, T/2, temporal_dim)
            
            # Stream B: Spatial texture from HHH + HLH
            f_spatial = self.spatial_stream(
                subbands['HHH'], subbands['HLH']
            )  # (B, T/2, spatial_dim)
            
            # Gated Cross-Merge: F_fused = F_spatial ⊙ σ(F_temporal)
            gate = torch.sigmoid(self.temporal_gate_proj(f_temporal))
            f_fused = f_spatial * gate
            
            # Project to output dim
            features = self.fusion_proj(f_fused)  # (B, T/2, out_dim)
            
        else:
            # V7.3 fallback: just pool HLL and project
            x = self.hll_proj(hll).squeeze(-1).squeeze(-1)  # (B, C, T/2)
            x = x.permute(0, 2, 1)  # (B, T/2, C)
            features = self.fallback_proj(x)  # (B, T/2, out_dim)
        
        return features, hll_energy
