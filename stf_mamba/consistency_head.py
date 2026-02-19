#!/usr/bin/env python3
"""
STF-Mamba V8.0 - Variance-Based Identity Consistency Head
===========================================================

Core detection signal: temporal identity variance.

For each clip:
    1. Compute per-frame cosine similarity to the sequence mean embedding
    2. Compute temporal variance of these similarities: σ²
    3. Concatenate: z = [mean_pool(H), σ²] → (B, 513)
    4. Classify: Linear(513 → 2) → logits

Signal interpretation:
    - Real faces → consistent identity across frames → LOW variance
    - Deepfakes → per-frame GAN drift → HIGH variance at blend boundaries

This is directly interpretable: high σ² = detected manipulation.
No need to learn "what a deepfake looks like" — learns "what instability looks like."

Input:  (B, T, 512) from Hydra-Mamba
Output: logits (B, 2), variance (B, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VarianceConsistencyHead(nn.Module):
    """
    Variance-based identity consistency classifier.

    Computes temporal identity variance as an explicit detection signal,
    then classifies based on both pooled features and the variance value.

    Args:
        embed_dim: Dimension of temporal embeddings from Hydra-Mamba.
            Default: 512.
        num_classes: Number of output classes. Default: 2 (real/fake).
        dropout: Dropout rate before classifier. Default: 0.1.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Classifier: [mean_pool (512) + variance (1)] → num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim + 1, num_classes),
        )

    def forward(self, h: torch.Tensor):
        """
        Args:
            h: (B, T, D) — temporally-aware embeddings from Hydra-Mamba.

        Returns:
            dict with:
                'logits': (B, 2) — classification logits.
                'variance': (B, 1) — temporal identity variance per clip.
                'similarities': (B, T) — per-frame cosine similarities.
        """
        # Step 1: Compute sequence mean embedding
        h_mean = h.mean(dim=1, keepdim=True)  # (B, 1, D)

        # Step 2: Per-frame cosine similarity to mean
        # Normalize both for cosine similarity
        h_norm = F.normalize(h, p=2, dim=-1)           # (B, T, D)
        h_mean_norm = F.normalize(h_mean, p=2, dim=-1)  # (B, 1, D)

        # Cosine similarity: dot product of normalized vectors
        similarities = (h_norm * h_mean_norm).sum(dim=-1)  # (B, T)

        # Step 3: Temporal variance of similarities
        variance = similarities.var(dim=1, keepdim=True)  # (B, 1)

        # Step 4: Concatenate mean-pooled features + variance
        h_pooled = h.mean(dim=1)  # (B, D) — mean pool over time
        z = torch.cat([h_pooled, variance], dim=-1)  # (B, D+1)

        # Step 5: Classify
        logits = self.classifier(z)  # (B, 2)

        return {
            "logits": logits,
            "variance": variance,
            "similarities": similarities,
        }

    def get_param_groups(self, lr_head: float = 1e-4):
        """Returns parameter groups for the optimizer."""
        return [{"params": self.parameters(), "lr": lr_head}]
