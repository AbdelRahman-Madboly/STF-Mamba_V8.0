#!/usr/bin/env python3
"""
STF-Mamba V8.0 - Modular Configuration
========================================

Extends V7.3 config with FEATURE FLAGS for incremental upgrades.
Train in phases:
  Phase A: V7.3 arch + fixed loss + augmentation → break 51% barrier  
  Phase B: Enable Hydra + W-SS + Mamer → full V8.0

Usage:
    # Phase A (safe, budget-friendly)
    config = STFV8Config.phase_a()
    
    # Phase B (full V8.0)
    config = STFV8Config.phase_b()
    
    # Custom
    config = STFV8Config(use_hydra=True, use_mamer=False, ...)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class STFV8Config:
    """
    STF-Mamba V8.0 configuration with progressive upgrade flags.
    
    ARCHITECTURE (when all flags enabled):
    ┌─────────────────────────────────────────────────────┐
    │  ConvNeXt V2-Base (frozen/low-LR)                   │
    │  ↓                                                   │
    │  3D-DWT (Sym2) → W-SS (Wavelet-Selective Scan)     │
    │  ↓                                                   │
    │  4× PN-Mamer Blocks (Hydra + Attention)             │
    │  ↓                                                   │
    │  Classification Head + Contrastive HLL Loss          │
    └─────────────────────────────────────────────────────┘
    """
    
    # ==========================================
    # FEATURE FLAGS (the upgrade switches)
    # ==========================================
    
    # Phase A fixes (safe, always enable these)
    use_contrastive_hll: bool = True       # Replace MSE(HLL,0) with contrastive loss
    use_compression_aug: bool = True       # Random H.264 re-compression during training
    use_multiscale_drift: bool = True      # Multi-scale Brownian drift (not just ±2px)
    use_hll_normalization: bool = True     # Normalize HLL relative to total band energy
    
    # Phase B upgrades (enable when Phase A works)
    use_hydra: bool = False                # Replace Bi-Mamba with Quasiseparable Hydra
    use_mamer: bool = False                # Replace FFN with Attention in Mamba blocks
    use_wss: bool = False                  # Wavelet-Selective Scan (frequency-native)
    use_adaptive_sdim: bool = False        # Dynamic state dim based on HLL energy
    use_pn_encoder: bool = False           # Parallel-Noncausal channel mixing
    
    # ==========================================
    # BACKBONE (kept from V7.3 — don't change what works)
    # ==========================================
    backbone_name: str = "convnextv2_base"  # Pre-trained spatial anchor
    backbone_pretrained: bool = True
    backbone_freeze_stages: int = 2         # Freeze first N stages (save memory + prevent overfitting)
    backbone_lr_mult: float = 0.01          # 100x lower LR for backbone
    backbone_out_dim: int = 1024            # ConvNeXt V2-Base output dimension
    
    # ==========================================
    # INPUT
    # ==========================================
    num_frames: int = 32
    img_size: int = 224                     # 224 not 256 — matches ConvNeXt pretraining resolution
    in_channels: int = 3
    num_classes: int = 2
    
    # ==========================================
    # 3D-DWT MODULE
    # ==========================================
    dwt_wavelet: str = "sym2"
    dwt_out_channels: int = 64
    
    # ==========================================
    # W-SS MODULE (Phase B)
    # ==========================================
    wss_temporal_dim: int = 128             # Dim for HLL temporal stream
    wss_spatial_dim: int = 128              # Dim for HHH/HLH spatial stream
    wss_bands_temporal: tuple = ("HLL",)    # Bands for temporal stream
    wss_bands_spatial: tuple = ("HHH", "HLH")  # Bands for spatial stream
    
    # ==========================================
    # HYDRA MIXER (Phase B)
    # ==========================================
    hydra_d_state: int = 64                 # SSM state dimension (Hydra default)
    hydra_d_conv: int = 7                   # Local conv width (±3 frames)
    hydra_mixer_type: str = "quasiseparable"
    
    # ==========================================
    # MAMBA (V7.3 fallback when Hydra disabled)
    # ==========================================
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    
    # ==========================================
    # PN-MAMER BLOCKS
    # ==========================================
    num_temporal_stages: int = 4            # Number of PN-Mamer (or Bi-Mamba) stages
    blocks_per_stage: int = 3               # Hydra blocks per stage (N in MamBo-3-Hydra-N3)
    mamer_num_heads: int = 8                # Attention heads in Mamer layer
    mamer_attention_every_n: int = 3        # Insert attention every N SSM blocks
    temporal_dim: int = 256                 # Feature dim for temporal modeling
    
    # ==========================================
    # ADAPTIVE SCALING (Phase B)
    # ==========================================
    adaptive_d_base: int = 16               # Base state dim
    adaptive_num_experts: int = 4           # MoE expert count
    adaptive_gamma_init: float = 1.0        # Learnable sensitivity
    
    # ==========================================
    # LOSSES
    # ==========================================
    lambda_var: float = 0.5                 # Weight for variance auxiliary loss
    variance_margin: float = 1.5            # Margin for log-variance ranking loss
    
    # ==========================================
    # REGULARIZATION
    # ==========================================
    dropout: float = 0.3
    drop_path: float = 0.1
    label_smoothing: float = 0.1
    
    # ==========================================
    # TRAINING
    # ==========================================
    # Compute budget
    max_epochs: int = 100
    batch_size: int = 4                     # RTX 4090: 4, A100: 8
    grad_accum_steps: int = 4               # Effective batch = 16
    
    # Optimizer
    lr: float = 5e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Augmentation
    compression_qp_range: tuple = (23, 40)  # H.264 QP range for compression aug
    drift_scale_range: tuple = (0.5, 5.0)   # Multi-scale Brownian drift (pixels)
    drift_use_tps: bool = True              # Thin-plate spline deformation
    
    # Memory optimization
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True         # fp16/bf16
    num_workers: int = 4
    pin_memory: bool = True
    
    # ==========================================
    # PATHS
    # ==========================================
    data_root: str = "/workspace/data"
    output_dir: str = "/workspace/outputs"
    checkpoint_dir: str = "/workspace/checkpoints"
    
    # ==========================================
    # CLASS METHODS FOR QUICK CONFIGS
    # ==========================================
    
    @classmethod
    def phase_a(cls, **overrides) -> 'STFV8Config':
        """
        Phase A: V7.3 architecture + critical fixes.
        Budget: ~$2.50 on RTX 4090 (~70 epochs)
        Expected: Break 51% AUC barrier → target 75-85% on real deepfakes
        """
        defaults = dict(
            # Phase A fixes ON
            use_contrastive_hll=True,
            use_compression_aug=True,
            use_multiscale_drift=True,
            use_hll_normalization=True,
            # Phase B OFF (save for later)
            use_hydra=False,
            use_mamer=False,
            use_wss=False,
            use_adaptive_sdim=False,
            use_pn_encoder=False,
            # Conservative training
            max_epochs=70,
            batch_size=4,
            lr=3e-4,
        )
        defaults.update(overrides)
        return cls(**defaults)
    
    @classmethod
    def phase_b(cls, **overrides) -> 'STFV8Config':
        """
        Phase B: Full V8.0 architecture.
        Budget: ~$2.50 on RTX 4090 (~50 epochs, fine-tune from Phase A checkpoint)
        Expected: 85-95% AUC on cross-dataset evaluation
        """
        defaults = dict(
            # ALL flags ON
            use_contrastive_hll=True,
            use_compression_aug=True,
            use_multiscale_drift=True,
            use_hll_normalization=True,
            use_hydra=True,
            use_mamer=True,
            use_wss=True,
            use_adaptive_sdim=True,
            use_pn_encoder=True,
            # Fine-tuning settings
            max_epochs=50,
            batch_size=4,
            lr=1e-4,  # Lower LR for fine-tuning
            warmup_epochs=3,
        )
        defaults.update(overrides)
        return cls(**defaults)
    
    @classmethod
    def debug(cls, **overrides) -> 'STFV8Config':
        """Minimal config for testing on CPU/small GPU."""
        defaults = dict(
            num_frames=8,
            img_size=112,
            batch_size=2,
            temporal_dim=64,
            backbone_out_dim=256,
            max_epochs=2,
            num_workers=0,
            use_mixed_precision=False,
        )
        defaults.update(overrides)
        return cls(**defaults)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize config for logging."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def describe_mode(self) -> str:
        """Human-readable description of current mode."""
        flags = []
        if self.use_contrastive_hll: flags.append("ContrastiveHLL")
        if self.use_compression_aug: flags.append("CompressionAug")
        if self.use_multiscale_drift: flags.append("MultiScaleDrift")
        if self.use_hll_normalization: flags.append("HLLNorm")
        if self.use_hydra: flags.append("Hydra")
        if self.use_mamer: flags.append("Mamer")
        if self.use_wss: flags.append("W-SS")
        if self.use_adaptive_sdim: flags.append("AdaptiveS")
        if self.use_pn_encoder: flags.append("PN-Enc")
        
        phase = "Phase B (Full V8.0)" if self.use_hydra else "Phase A (V7.3 + Fixes)"
        return f"{phase}: [{', '.join(flags)}]"
    
    def estimate_cost(self, gpu_name: str = "RTX 4090") -> dict:
        """Estimate training cost."""
        # Rough estimates (seconds per epoch)
        gpu_speeds = {
            "RTX 4090": {"base": 280, "v8": 380, "cost_hr": 0.39},
            "A40": {"base": 300, "v8": 400, "cost_hr": 0.79},
            "A100 80GB": {"base": 234, "v8": 328, "cost_hr": 1.64},
        }
        
        if gpu_name not in gpu_speeds:
            return {"error": f"Unknown GPU: {gpu_name}"}
        
        gpu = gpu_speeds[gpu_name]
        is_v8 = self.use_hydra or self.use_mamer
        sec_per_epoch = gpu["v8"] if is_v8 else gpu["base"]
        total_sec = sec_per_epoch * self.max_epochs
        total_hrs = total_sec / 3600
        total_cost = total_hrs * gpu["cost_hr"]
        
        return {
            "gpu": gpu_name,
            "epochs": self.max_epochs,
            "sec_per_epoch": sec_per_epoch,
            "total_hours": round(total_hrs, 2),
            "total_cost_usd": round(total_cost, 2),
        }
