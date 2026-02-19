"""
STF-Mamba V8.0 - Semantic Temporal Forensics via Hydra-Mamba and DINOv2
========================================================================

Deepfake video detection through semantic identity state consistency.

Architecture:
    DINOv2-ViT-B/14 → Temporal Projection → Hydra-Mamba x2 → Variance Identity Head

Target: CVPR/ICCV 2026 | Celeb-DF AUC >= 0.90
"""

__version__ = "8.0.0"
__author__ = "Abdel Rahman Madboly"

from stf_mamba.backbone import DINOv2Backbone
from stf_mamba.hydra_mamba import HydraMambaTemporalModule, is_mamba_available
from stf_mamba.consistency_head import VarianceConsistencyHead
from stf_mamba.model import STFMambaV8
from stf_mamba.losses import STFMambaLoss

__all__ = [
    "DINOv2Backbone",
    "HydraMambaTemporalModule",
    "VarianceConsistencyHead",
    "STFMambaV8",
    "STFMambaLoss",
    "is_mamba_available",
]
