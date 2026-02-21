#!/usr/bin/env python3
"""
STF-Mamba V8.0 - Full Model Assembly
======================================

Assembles the complete V8.0 detection pipeline:

    DINOv2-ViT-B/14 → Temporal Projection → Hydra-Mamba x2 → Variance Consistency Head

Input:  (B, T, 3, 224, 224) — T face-cropped frames per clip
Output: {'logits': (B, 2), 'variance': (B, 1), 'similarities': (B, T)}

Architecture decisions (all empirically justified):
    - DINOv2 backbone: semantic features that survive H.264 compression
    - Partial freeze (blocks 0-9): preserves pretrained knowledge
    - Hydra-Mamba x2: bidirectional identity state modeling
    - Variance head: detects per-frame GAN identity drift
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from stf_mamba.backbone import DINOv2Backbone
from stf_mamba.hydra_mamba import HydraMambaTemporalModule
from stf_mamba.consistency_head import VarianceConsistencyHead

logger = logging.getLogger(__name__)


class STFMambaV8(nn.Module):
    """
    STF-Mamba V8.0 — Semantic Temporal Forensics Deepfake Detector.

    Args:
        backbone_name: DINOv2 model variant. Default: 'dinov2_vitb14'.
        freeze_blocks: DINOv2 blocks to freeze. Default: 10.
        proj_dim: Temporal projection dimension. Default: 512.
        num_temporal_blocks: Stacked Hydra-Mamba blocks. Default: 2.
        d_state: SSM state dimension. Default: 64.
        d_conv: SSM convolution width. Default: 7.
        expand: SSM expansion factor. Default: 2.
        dropout: Dropout rate. Default: 0.1.
        num_classes: Output classes. Default: 2.
        pretrained_backbone: Load pretrained DINOv2. Default: True.
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_vitb14",
        freeze_blocks: int = 10,
        proj_dim: int = 512,
        num_temporal_blocks: int = 2,
        d_state: int = 64,
        d_conv: int = 7,
        expand: int = 2,
        dropout: float = 0.1,
        num_classes: int = 2,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()

        # Stage 1: DINOv2 Backbone
        self.backbone = DINOv2Backbone(
            model_name=backbone_name,
            freeze_blocks=freeze_blocks,
            pretrained=pretrained_backbone,
        )

        # Stage 2+3: Temporal Projection + Hydra-Mamba
        self.temporal = HydraMambaTemporalModule(
            input_dim=DINOv2Backbone.CLS_DIM,  # 768
            proj_dim=proj_dim,
            num_blocks=num_temporal_blocks,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        # Stage 4: Variance Consistency Head
        self.head = VarianceConsistencyHead(
            embed_dim=proj_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        # Log total params
        total, trainable = self._count_params()
        logger.info(
            f"STFMambaV8: {total / 1e6:.1f}M total, "
            f"{trainable / 1e6:.1f}M trainable"
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            x: (B, T, 3, 224, 224) — batch of T-frame video clips.

        Returns:
            dict with:
                'logits': (B, 2) — real/fake classification.
                'variance': (B, 1) — Log-scaled temporal identity variance.
                'similarities': (B, T) — per-frame cosine similarities (for logging).
        """
        # 1. DINOv2: (B, T, 3, 224, 224) → (B, T, 768)
        cls_tokens = self.backbone(x)

        # 2. Hydra-Mamba: (B, T, 768) → (B, T, 512)
        temporal_features = self.temporal(cls_tokens)

        # 3. Consistency Head: (B, T, 512) → logits + log_variance
        # The updated head now computes variance in a learned projection space.
        output = self.head(temporal_features)

        return output

    def get_param_groups(
        self,
        lr_backbone: float = 5e-6,
        lr_temporal: float = 1e-4,
        lr_head: float = 1e-4,
    ) -> List[Dict]:
        """
        Returns differential learning rate parameter groups.

        Args:
            lr_backbone: LR for DINOv2 fine-tuned blocks (10-11).
            lr_temporal: LR for Hydra-Mamba + projection.
            lr_head: LR for consistency head.

        Returns:
            List of param group dicts for the optimizer.
        """
        groups = []
        groups.extend(self.backbone.get_param_groups(lr_backbone))
        groups.extend(self.temporal.get_param_groups(lr_temporal))
        groups.extend(self.head.get_param_groups(lr_head))
        return groups

    def _count_params(self):
        """Returns (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    @classmethod
    def from_config(cls, config: dict) -> "STFMambaV8":
        """
        Build model from a YAML config dict.

        Args:
            config: Parsed YAML config with 'model' section.

        Returns:
            Initialized STFMambaV8 model.
        """
        m = config["model"]
        h = m["hydra"]
        return cls(
            backbone_name=m["backbone"],
            freeze_blocks=m["freeze_blocks"],
            proj_dim=m["temporal_proj_dim"],
            num_temporal_blocks=m["temporal_num_blocks"],
            d_state=h["d_state"],
            d_conv=h["d_conv"],
            expand=h["expand"],
            dropout=h["dropout"],
            num_classes=m["consistency_head"]["num_classes"],
        )
