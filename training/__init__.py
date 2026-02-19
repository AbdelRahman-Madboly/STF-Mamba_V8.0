"""
STF-Mamba V8.0 - Training Infrastructure
==========================================

Components:
    - optimizer: AdamW with differential LRs + gradient clipping
    - scheduler: Linear warmup + cosine annealing
    - trainer: Full training loop with AUC validation + checkpointing
"""

from training.optimizer import build_optimizer, clip_gradients
from training.scheduler import build_scheduler
from training.trainer import Trainer

__all__ = [
    "build_optimizer",
    "clip_gradients",
    "build_scheduler",
    "Trainer",
]
