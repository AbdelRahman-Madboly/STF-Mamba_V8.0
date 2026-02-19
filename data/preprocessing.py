"""
STF-Mamba V8.0 - Face Preprocessing
======================================

dlib 81-point landmark detection -> face bounding box -> 1.3x expand -> 224x224 crop.
All crops pre-cached as NPZ: crops/{video_id}.npz
Landmarks pre-cached as NPZ: landmarks/{video_id}.npz
"""

# TODO: Stage 3 implementation
