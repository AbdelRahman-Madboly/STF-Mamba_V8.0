"""
STF-Mamba V8.0 - SBI Dataset
==============================

Self-Blended Images dataset with pre-caching.
Reads official splits, generates SBI fakes via Poisson blending + Brownian drift mask.

CRITICAL: All SBI fakes are pre-cached to disk (NPZ).
On-the-fly SBI without caching causes contradictory gradients (proven in V7.3).

Returns: {'frames': (T, 3, H, W), 'label': 0 or 1, 'variance_gt': float}
"""

# TODO: Stage 3 implementation
