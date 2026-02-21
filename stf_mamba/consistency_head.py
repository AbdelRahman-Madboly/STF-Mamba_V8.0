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

        # Learned projection to find "identity drift" dimensions
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.LeakyReLU(0.1),
            nn.Linear(embed_dim // 4, embed_dim // 8)
        )

        # Classifier: [mean_pool (512) + log_variance (1)] → num_classes
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
        # Step 1: Project to a lower-dimensional drift space
        # This amplifies subtle flicker that global pooling ignores
        h_proj = self.projection(h)  # (B, T, D')

        # Step 2: Compute Log-Variance for numerical stability
        # Raw variance of stable features is often < 1e-5
        # Log-scaling makes the gradient signal visible to AdamW
        var_per_dim = h_proj.var(dim=1)  # (B, D')
        log_var = torch.log(var_per_dim + 1e-6).mean(dim=-1, keepdim=True)  # (B, 1)

        # Step 3: Compute original similarities for logging only
        with torch.no_grad():
            h_norm = F.normalize(h, p=2, dim=-1)
            h_mean_norm = F.normalize(h.mean(dim=1, keepdim=True), p=2, dim=-1)
            similarities = (h_norm * h_mean_norm).sum(dim=-1)

        # Step 4: Concatenate mean-pooled features + variance
        h_pooled = h.mean(dim=1)  # (B, D)
        z = torch.cat([h_pooled, log_var], dim=-1)  # (B, D+1)

        # Step 5: Classify
        logits = self.classifier(z)  # (B, 2)

        return {
            "logits": logits,
            "variance": log_var,  # Passing log_var to the loss function
            "similarities": similarities,
        }

    def get_param_groups(self, lr_head: float = 1e-4):
        """Returns parameter groups for the optimizer."""
        return [{"params": self.parameters(), "lr": lr_head}]
