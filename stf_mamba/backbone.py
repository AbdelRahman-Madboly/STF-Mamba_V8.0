#!/usr/bin/env python3
"""
STF-Mamba V8.0 - Main Backbone Model
=======================================

Modular architecture with feature flags for incremental upgrades.

PHASE A (V7.3 + Fixes):
  ConvNeXt V2 → 3D-DWT → Simple HLL features → Bi-Mamba + FFN → Head
  + Contrastive HLL Loss (fix #1)
  + Compression Augmentation (fix #2)
  + Multi-scale Drift (fix #3)

PHASE B (Full V8.0):
  ConvNeXt V2 → 3D-DWT → W-SS (dual stream) → PN-Mamer (Hydra + Attention) → Head

Usage:
    from config_v8 import STFV8Config
    from modules.backbone_v8 import STFMambaV8
    
    # Phase A
    model = STFMambaV8(STFV8Config.phase_a())
    
    # Phase B  
    model = STFMambaV8(STFV8Config.phase_b())
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class STFMambaV8(nn.Module):
    """
    STF-Mamba V8.0: Unified Spectral-Temporal Forensics.
    
    Architecture:
    1. ConvNeXt V2-Base (pretrained, frozen/low-LR) → frame-level spatial features
    2. 3D-DWT (Sym2) → 8 wavelet sub-bands 
    3. W-SS or simple HLL (depending on config) → frequency features
    4. PN-Mamer stages (Hydra/Bi-Mamba + Attention/FFN) → temporal modeling
    5. Classification head → real/fake prediction
    
    Returns dict with logits, hll_energy, all_band_energy for loss computation.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # =====================================================
        # PHASE 1: Spatial Feature Extraction (ConvNeXt V2)
        # =====================================================
        self.backbone = self._build_backbone(config)
        backbone_dim = config.backbone_out_dim
        
        # =====================================================
        # PHASE 2: 3D-DWT Frequency Decomposition
        # =====================================================
        self.dwt = DWT3DModule(in_channels=backbone_dim, out_channels=config.dwt_out_channels)
        
        # =====================================================
        # PHASE 2.5: W-SS or Simple Frequency Feature Extraction
        # =====================================================
        from .wss_module import WaveletSelectiveScan
        self.wss = WaveletSelectiveScan(
            in_channels=config.dwt_out_channels,
            out_dim=config.temporal_dim,
            temporal_dim=config.wss_temporal_dim,
            spatial_dim=config.wss_spatial_dim,
            dropout=config.dropout,
            use_wss=config.use_wss,
        )
        
        # =====================================================
        # PHASE 3: Temporal Modeling (PN-Mamer or Bi-Mamba)
        # =====================================================
        from .hydra_mixer import PNMamerStage
        self.temporal_stages = nn.ModuleList()
        for i in range(config.num_temporal_stages):
            stage = PNMamerStage(
                d_model=config.temporal_dim,
                num_hydra_blocks=config.blocks_per_stage,
                num_heads=config.mamer_num_heads,
                d_state=config.hydra_d_state if config.use_hydra else config.mamba_d_state,
                d_conv=config.hydra_d_conv if config.use_hydra else config.mamba_d_conv,
                dropout=config.dropout,
                use_hydra=config.use_hydra,
                use_mamer=config.use_mamer,
            )
            self.temporal_stages.append(stage)
        
        # =====================================================
        # PHASE 4: Classification Head
        # =====================================================
        self.head_norm = nn.LayerNorm(config.temporal_dim)
        self.head = nn.Sequential(
            nn.Linear(config.temporal_dim, config.temporal_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.temporal_dim // 2, config.num_classes),
        )
        
        # Initialize
        self._init_weights()
        
        # Log architecture info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"STF-Mamba V8.0 initialized: {config.describe_mode()}")
        logger.info(f"  Total params: {total_params:,}")
        logger.info(f"  Trainable params: {trainable_params:,}")
        
    def _build_backbone(self, config):
        """Build and configure ConvNeXt V2 backbone."""
        try:
            import timm
            backbone = timm.create_model(
                config.backbone_name,
                pretrained=config.backbone_pretrained,
                num_classes=0,  # Remove classification head
                global_pool='',  # We'll do our own pooling
            )
            
            # Freeze early stages
            if config.backbone_freeze_stages > 0:
                # Freeze stem
                for param in backbone.stem.parameters():
                    param.requires_grad = False
                # Freeze stages
                for i in range(min(config.backbone_freeze_stages, 4)):
                    if hasattr(backbone, 'stages'):
                        for param in backbone.stages[i].parameters():
                            param.requires_grad = False
                            
            logger.info(f"Loaded {config.backbone_name} (pretrained={config.backbone_pretrained})")
            logger.info(f"  Frozen stages: {config.backbone_freeze_stages}")
            
            return backbone
            
        except Exception as e:
            logger.warning(f"Could not load {config.backbone_name}: {e}")
            logger.warning("Using lightweight CNN fallback")
            return LightweightCNNBackbone(
                in_channels=config.in_channels,
                out_dim=config.backbone_out_dim,
            )
    
    def _extract_frame_features(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract per-frame features using ConvNeXt V2.
        
        Args:
            video: (B, C, T, H, W) raw video
        Returns:
            features: (B, T, backbone_dim) per-frame feature vectors
        """
        B, C, T, H, W = video.shape
        
        # Reshape to process all frames as a batch: (B*T, C, H, W)
        frames = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        
        # Extract features
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            feat = self.backbone(frames)  # (B*T, backbone_dim, h, w) or (B*T, backbone_dim)
            
            # Handle different backbone output formats
            if feat.dim() == 4:
                feat = feat.mean(dim=[-2, -1])  # Global average pool → (B*T, backbone_dim)
            elif feat.dim() == 3:
                feat = feat.mean(dim=1)  # Pool sequence dim
        
        # Reshape back: (B, T, backbone_dim)
        features = feat.reshape(B, T, -1)
        
        return features
    
    def forward(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            video: (B, C, T, H, W) input video tensor
            
        Returns:
            Dict with:
                - logits: (B, num_classes) classification logits
                - hll_energy: (B,) HLL energy for contrastive loss
                - all_band_energy: (B,) total band energy for normalization
        """
        B = video.shape[0]
        
        # === Phase 1: Spatial features ===
        frame_features = self._extract_frame_features(video)  # (B, T, backbone_dim)
        
        # === Phase 2: 3D-DWT decomposition ===
        # Reshape for 3D-DWT: (B, T, D) → (B, D, T, 1, 1) → apply DWT
        subbands, all_band_energy = self.dwt(frame_features)
        
        # === Phase 2.5: Frequency feature extraction (W-SS or simple) ===
        freq_features, hll_energy = self.wss(subbands)  # (B, T/2, temporal_dim)
        
        # === Phase 3: Temporal modeling ===
        x = freq_features
        for stage in self.temporal_stages:
            if self.config.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
        
        # === Phase 4: Classification ===
        x = self.head_norm(x)
        x = x.mean(dim=1)  # Global average pool over time
        logits = self.head(x)
        
        return {
            'logits': logits,
            'hll_energy': hll_energy,
            'all_band_energy': all_band_energy,
        }
    
    def _init_weights(self):
        """Initialize non-pretrained weights."""
        for name, m in self.named_modules():
            if 'backbone' in name:
                continue  # Skip pretrained backbone
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def get_optimizer_param_groups(self) -> list:
        """
        Get parameter groups with differential learning rates.
        
        Backbone: config.lr * config.backbone_lr_mult (very low)
        Rest: config.lr (normal)
        """
        backbone_params = []
        other_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
        
        return [
            {'params': other_params, 'lr': self.config.lr},
            {'params': backbone_params, 'lr': self.config.lr * self.config.backbone_lr_mult},
        ]


# =====================================================
# DWT MODULE (adapted from V7.3)
# =====================================================

class DWT3DModule(nn.Module):
    """
    Simplified 3D-DWT for sequence features.
    
    Takes per-frame features (B, T, D) and produces wavelet sub-bands
    along the temporal dimension.
    
    For V8.0, we apply 1D DWT along time (since spatial features are 
    already extracted by ConvNeXt V2), producing 2 sub-bands: L and H.
    We then create pseudo-3D sub-bands by combining temporal DWT with 
    feature-space decomposition.
    """
    
    def __init__(self, in_channels: int = 1024, out_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Project backbone features to DWT working dimension
        self.proj = nn.Linear(in_channels, out_channels * 4)
        
        # Learnable wavelet-like temporal decomposition
        # Low-pass (smooth temporal trends)
        self.temporal_low = nn.Conv1d(
            out_channels * 4, out_channels, kernel_size=4, stride=2, padding=1,
            groups=min(out_channels, out_channels * 4),
        )
        # High-pass (temporal flicker) — this is our "HLL equivalent"
        self.temporal_high = nn.Conv1d(
            out_channels * 4, out_channels, kernel_size=4, stride=2, padding=1,
            groups=min(out_channels, out_channels * 4),
        )
        
        # Feature-space decomposition (simulates spatial frequency bands)
        self.band_projections = nn.ModuleDict({
            'HLL': nn.Linear(out_channels, out_channels),  # Temporal flicker
            'HHH': nn.Linear(out_channels, out_channels),  # High-freq spatial
            'HLH': nn.Linear(out_channels, out_channels),  # Mixed
            'LLL': nn.Linear(out_channels, out_channels),  # Smooth baseline
        })
        
    def forward(self, frame_features: torch.Tensor):
        """
        Args:
            frame_features: (B, T, backbone_dim) per-frame features
            
        Returns:
            subbands: Dict of sub-band tensors, each (B, out_channels, T/2, 1, 1)
            all_band_energy: (B,) total energy across all bands
        """
        B, T, D = frame_features.shape
        
        # Project to working dim
        x = self.proj(frame_features)  # (B, T, out_channels*4)
        
        # Temporal decomposition via Conv1d
        x_t = x.transpose(1, 2)  # (B, out_channels*4, T)
        
        low = self.temporal_low(x_t)    # (B, out_channels, T/2)
        high = self.temporal_high(x_t)  # (B, out_channels, T/2)
        
        T_half = low.shape[2]
        
        # Create sub-bands via feature-space projection
        high_seq = high.transpose(1, 2)  # (B, T/2, out_channels)
        low_seq = low.transpose(1, 2)
        
        subbands = {}
        for band_name, proj in self.band_projections.items():
            if band_name.startswith('H'):
                # High temporal bands come from temporal_high
                band = proj(high_seq)  # (B, T/2, out_channels)
            else:
                # Low temporal bands come from temporal_low
                band = proj(low_seq)
            # Reshape to pseudo-5D: (B, out_channels, T/2, 1, 1)
            subbands[band_name] = band.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        
        # Compute total band energy for normalization
        total_energy = torch.zeros(B, device=frame_features.device)
        for band_tensor in subbands.values():
            total_energy += band_tensor.reshape(B, -1).norm(dim=1) ** 2
        all_band_energy = total_energy.sqrt()
        
        return subbands, all_band_energy


# =====================================================
# FALLBACK BACKBONE (when timm not available)
# =====================================================

class LightweightCNNBackbone(nn.Module):
    """Fallback when ConvNeXt V2 not available."""
    
    def __init__(self, in_channels: int = 3, out_dim: int = 1024):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, out_dim),
        )
    
    def forward(self, x):
        return self.features(x)
