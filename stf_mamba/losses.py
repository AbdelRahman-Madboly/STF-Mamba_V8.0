#!/usr/bin/env python3
"""
STF-Mamba V8.0 - Loss Functions
=================================

The #1 fix: Replace MSE(HLL_real, 0) with Contrastive HLL Loss.

WHY THIS MATTERS:
- V7.3 used L_HLL = MSE(HLL_real, 0) which teaches "real videos have zero temporal flicker"
- But real videos DO have non-zero HLL energy from natural motion, lighting, compression
- The model resolves this contradiction by learning dataset-specific shortcuts
- Contrastive loss instead teaches "fakes have MORE flicker than reals" (relative, not absolute)

LOSS COMPONENTS:
1. CrossEntropy with label smoothing (classification)
2. Contrastive HLL Loss (frequency regularization)  
3. Optional: HLL Normalization (compression invariance)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ContrastiveHLLLoss(nn.Module):
    """
    Contrastive HLL Loss: Margin-based separation of real vs fake HLL energy.
    
    Instead of: MSE(HLL_real, 0)  — "real = zero flicker" (WRONG)
    We use:     max(0, m - (||HLL_fake|| - ||HLL_real||))  — "fake > real by margin m"
    
    This is domain-invariant because:
    - A noisy DFDC video has high HLL for BOTH real and fake → margin still holds
    - A clean FF++ video has low HLL for both → margin still holds
    - Only the RELATIVE difference matters, not absolute values
    """
    
    def __init__(self, margin: float = 1.0, normalize: bool = True):
        """
        Args:
            margin: Minimum desired separation between fake and real HLL energy
            normalize: Whether to normalize HLL energy relative to total band energy
        """
        super().__init__()
        self.margin = margin
        self.normalize = normalize
        
    def forward(
        self, 
        hll_energy: torch.Tensor,  # (B,) or (B, T) — HLL L2 energy per sample
        labels: torch.Tensor,       # (B,) — 0=real, 1=fake
        all_band_energy: Optional[torch.Tensor] = None  # (B,) — total DWT energy for normalization
    ) -> torch.Tensor:
        """
        Compute contrastive HLL loss.
        
        Args:
            hll_energy: L2 norm of HLL sub-band per sample
            labels: Binary labels (0=real, 1=fake)  
            all_band_energy: Total energy across all 8 DWT sub-bands (for normalization)
            
        Returns:
            Scalar loss
        """
        # Normalize HLL relative to total band energy (compression invariance)
        if self.normalize and all_band_energy is not None:
            hll_energy = hll_energy / (all_band_energy + 1e-8)
        
        # Separate real and fake
        real_mask = (labels == 0)
        fake_mask = (labels == 1)
        
        # Need at least one of each in the batch
        if real_mask.sum() == 0 or fake_mask.sum() == 0:
            return torch.tensor(0.0, device=hll_energy.device, requires_grad=True)
        
        real_energy = hll_energy[real_mask].mean()
        fake_energy = hll_energy[fake_mask].mean()
        
        # Margin loss: we want fake_energy - real_energy >= margin
        loss = F.relu(self.margin - (fake_energy - real_energy))
        
        return loss


class InfoNCEHLLLoss(nn.Module):
    """
    InfoNCE-style HLL loss for scale-invariant separation.
    
    L = -log(exp(||HLL_fake|| / τ) / (exp(||HLL_fake|| / τ) + exp(||HLL_real|| / τ)))
    
    This naturally learns the separation in a scale-invariant way.
    More stable than margin loss when HLL energy ranges vary across batches.
    """
    
    def __init__(self, temperature: float = 0.1, normalize: bool = True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        
    def forward(
        self, 
        hll_energy: torch.Tensor,
        labels: torch.Tensor,
        all_band_energy: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.normalize and all_band_energy is not None:
            hll_energy = hll_energy / (all_band_energy + 1e-8)
            
        real_mask = (labels == 0)
        fake_mask = (labels == 1)
        
        if real_mask.sum() == 0 or fake_mask.sum() == 0:
            return torch.tensor(0.0, device=hll_energy.device, requires_grad=True)
        
        real_energy = hll_energy[real_mask].mean()
        fake_energy = hll_energy[fake_mask].mean()
        
        # InfoNCE: probability that fake has higher energy than real
        logits = torch.stack([fake_energy, real_energy]) / self.temperature
        # Target: index 0 (fake should be the "positive" = higher energy)
        target = torch.tensor(0, device=logits.device)
        loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
        
        return loss


class STFV8Loss(nn.Module):
    """
    Complete V8.0 loss function.
    
    L_total = L_CE + λ * L_HLL_contrastive
    
    Replaces V7.3's: L_total = L_CE + λ * MSE(HLL_real, 0)
    """
    
    def __init__(
        self,
        hll_weight: float = 0.1,
        hll_margin: float = 1.0,
        hll_temperature: float = 0.1,
        hll_mode: str = "margin",       # "margin" or "infonce"
        label_smoothing: float = 0.1,
        normalize_hll: bool = True,
    ):
        """
        Args:
            hll_weight: λ weight for HLL contrastive loss
            hll_margin: Margin for contrastive loss (if mode="margin")
            hll_temperature: Temperature for InfoNCE (if mode="infonce")
            hll_mode: "margin" for margin-based, "infonce" for InfoNCE
            label_smoothing: Label smoothing for CE loss
            normalize_hll: Normalize HLL energy by total band energy
        """
        super().__init__()
        self.hll_weight = hll_weight
        self.normalize_hll = normalize_hll
        
        # Classification loss
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # HLL contrastive loss
        if hll_mode == "margin":
            self.hll_loss = ContrastiveHLLLoss(
                margin=hll_margin, normalize=normalize_hll
            )
        elif hll_mode == "infonce":
            self.hll_loss = InfoNCEHLLLoss(
                temperature=hll_temperature, normalize=normalize_hll
            )
        else:
            raise ValueError(f"Unknown hll_mode: {hll_mode}")
        
        # V7.3 fallback (for comparison/ablation)
        self.hll_loss_v73 = None  # Set externally if needed
        
    def forward(
        self,
        logits: torch.Tensor,           # (B, 2) classification logits
        labels: torch.Tensor,            # (B,) binary labels
        hll_energy: torch.Tensor,        # (B,) HLL sub-band energy
        all_band_energy: Optional[torch.Tensor] = None,  # (B,) total DWT energy
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        
        Returns dict with individual loss components for logging.
        """
        # Classification loss
        loss_ce = self.ce_loss(logits, labels)
        
        # HLL contrastive loss
        loss_hll = self.hll_loss(hll_energy, labels, all_band_energy)
        
        # Total
        loss_total = loss_ce + self.hll_weight * loss_hll
        
        return {
            'loss': loss_total,
            'loss_ce': loss_ce.detach(),
            'loss_hll': loss_hll.detach(),
            'hll_real_mean': hll_energy[labels == 0].mean().detach() if (labels == 0).any() else torch.tensor(0.0),
            'hll_fake_mean': hll_energy[labels == 1].mean().detach() if (labels == 1).any() else torch.tensor(0.0),
        }


def compute_hll_energy(
    dwt_subbands: Dict[str, torch.Tensor], 
    normalize: bool = True
) -> tuple:
    """
    Compute HLL energy and total band energy from DWT sub-bands.
    
    Args:
        dwt_subbands: Dict with keys like 'HLL', 'HHH', etc.
                      Each tensor shape: (B, C, T/2, H/2, W/2)
        normalize: Whether to compute total band energy for normalization
        
    Returns:
        hll_energy: (B,) L2 norm of HLL sub-band per sample
        all_band_energy: (B,) total L2 norm across all sub-bands (or None)
    """
    hll = dwt_subbands['HLL']  # (B, C, T/2, H/2, W/2)
    B = hll.shape[0]
    
    # L2 norm per sample: flatten all dims except batch
    hll_energy = hll.reshape(B, -1).norm(dim=1)  # (B,)
    
    all_band_energy = None
    if normalize:
        total = torch.zeros(B, device=hll.device)
        for band_name, band_tensor in dwt_subbands.items():
            total += band_tensor.reshape(B, -1).norm(dim=1) ** 2
        all_band_energy = total.sqrt()  # (B,)
    
    return hll_energy, all_band_energy
