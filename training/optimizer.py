"""
STF-Mamba V8.0 - Optimizer with Differential Learning Rates
==============================================================

Parameter groups:
  - frozen_params: lr=0 (DINOv2 blocks 0-9)
  - backbone_params: lr=5e-6 (DINOv2 blocks 10-11)
  - temporal_params: lr=1e-4 (Hydra-Mamba + projection)
  - head_params: lr=1e-4 (consistency head)

Optimizer: AdamW, weight_decay=1e-4
"""

# TODO: Stage 4 implementation
