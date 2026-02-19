#!/usr/bin/env python3
"""
STF-Mamba V8.0 - DINOv2-ViT-B/14 Backbone
============================================

Extracts per-frame CLS tokens from DINOv2-ViT-B/14.

Freeze strategy:
    - Blocks 0-9: Frozen (lr=0), preserves pretrained knowledge
    - Blocks 10-11: Fine-tuned (lr=5e-6), adapts to deepfake domain

Input:  (B, T, 3, 224, 224) — T face-cropped frames
Output: (B, T, 768) — CLS token per frame

Why DINOv2 over EfficientNet-B4:
    1. Self-supervised training on 142M images → features survive H.264 compression
    2. ViT CLS tokens encode holistic face identity without spatial bias
    3. 2026 reviewers expect SOTA backbones; EffNet-B4 was the 2022 standard
"""

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DINOv2Backbone(nn.Module):
    """
    DINOv2-ViT-B/14 backbone with partial freeze for deepfake detection.

    Processes T frames independently (shared weights), extracts CLS token
    per frame. Batched forward pass: reshape (B*T, 3, 224, 224) → forward
    → reshape (B, T, 768).

    Args:
        model_name: DINOv2 model variant. Default: 'dinov2_vitb14'.
        freeze_blocks: Number of transformer blocks to freeze (0-indexed).
            Default: 10 (freeze blocks 0-9, fine-tune 10-11).
        pretrained: Whether to load pretrained weights. Default: True.
    """

    CLS_DIM = 768  # DINOv2-ViT-B/14 CLS token dimension
    NUM_BLOCKS = 12  # Total transformer blocks in ViT-B

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        freeze_blocks: int = 10,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.freeze_blocks = freeze_blocks

        # Load DINOv2 via torch.hub
        if pretrained:
            logger.info(f"Loading {model_name} via torch.hub (pretrained=True)...")
            self.dinov2 = torch.hub.load(
                "facebookresearch/dinov2",
                model_name,
                pretrained=True,
            )
        else:
            # Should NEVER be used — random init is empirically disproven (V7.3)
            logger.warning(
                "Loading DINOv2 WITHOUT pretrained weights. "
                "This is empirically proven to fail — use pretrained=True."
            )
            self.dinov2 = torch.hub.load(
                "facebookresearch/dinov2",
                model_name,
                pretrained=False,
            )

        # Apply freeze strategy
        self._apply_freeze(freeze_blocks)

        # Log parameter counts
        total, trainable = self._count_params()
        logger.info(
            f"DINOv2 backbone: {total / 1e6:.1f}M total, "
            f"{trainable / 1e6:.1f}M trainable "
            f"(blocks {freeze_blocks}-{self.NUM_BLOCKS - 1} fine-tuned)"
        )

    def _apply_freeze(self, freeze_blocks: int) -> None:
        """Freeze patch embed, pos embed, cls token, and first N blocks."""
        # Freeze patch embedding
        for param in self.dinov2.patch_embed.parameters():
            param.requires_grad = False

        # Freeze cls_token and pos_embed
        if hasattr(self.dinov2, "cls_token"):
            self.dinov2.cls_token.requires_grad = False
        if hasattr(self.dinov2, "pos_embed"):
            self.dinov2.pos_embed.requires_grad = False

        # Freeze first N blocks
        for i in range(min(freeze_blocks, len(self.dinov2.blocks))):
            for param in self.dinov2.blocks[i].parameters():
                param.requires_grad = False

        # Ensure last blocks are trainable
        for i in range(freeze_blocks, len(self.dinov2.blocks)):
            for param in self.dinov2.blocks[i].parameters():
                param.requires_grad = True

        # Freeze norm layer (part of pretrained representation)
        if hasattr(self.dinov2, "norm"):
            for param in self.dinov2.norm.parameters():
                param.requires_grad = False

    def _count_params(self) -> Tuple[int, int]:
        """Returns (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_param_groups(
        self, lr_backbone: float = 5e-6
    ) -> List[Dict]:
        """
        Returns parameter groups for differential learning rates.

        Args:
            lr_backbone: Learning rate for fine-tuned blocks (10-11).

        Returns:
            List of param group dicts for the optimizer.
            Frozen params are excluded (they have requires_grad=False).
        """
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if not trainable_params:
            return []
        return [{"params": trainable_params, "lr": lr_backbone}]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CLS tokens from T frames in a single batched pass.

        Args:
            x: (B, T, 3, 224, 224) — batch of T-frame video clips.

        Returns:
            (B, T, 768) — CLS token per frame.
        """
        B, T, C, H, W = x.shape

        # Reshape to process all frames in one pass: (B*T, 3, 224, 224)
        x_flat = x.reshape(B * T, C, H, W)

        # DINOv2 forward — extract CLS token
        # forward_features returns the CLS token (first token after norm)
        cls_tokens = self._extract_cls(x_flat)  # (B*T, 768)

        # Reshape back to temporal sequence: (B, T, 768)
        cls_tokens = cls_tokens.reshape(B, T, -1)

        return cls_tokens

    def _extract_cls(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CLS token from DINOv2.

        DINOv2's forward_features returns class_token by default.
        We handle both the standard and non-standard cases.

        Args:
            x: (N, 3, 224, 224) — batch of images.

        Returns:
            (N, 768) — CLS tokens.
        """
        # Use DINOv2's built-in feature extraction
        # This runs: patch_embed → pos_embed → blocks → norm → cls_token
        features = self.dinov2.forward_features(x)

        # DINOv2 returns a dict with 'x_norm_clstoken' key
        if isinstance(features, dict):
            return features["x_norm_clstoken"]

        # Fallback: if it returns raw tensor, take the CLS token (index 0)
        return features[:, 0]

    @torch.no_grad()
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """
        Enable gradient checkpointing to reduce VRAM usage on T4.

        Trades compute for memory — useful when batch_size=8 + 32 frames
        approaches the 15GB T4 VRAM limit.
        """
        for block in self.dinov2.blocks:
            block.use_checkpoint = enable
        state = "enabled" if enable else "disabled"
        logger.info(f"Gradient checkpointing {state} for DINOv2 blocks")
