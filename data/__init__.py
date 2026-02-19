"""
STF-Mamba V8.0 - Data Pipeline
=================================

Components:
    - splits: Official FF++ split loader (train/val/test JSON)
    - preprocessing: dlib face crop + NPZ caching
    - augmentation: DINOv2-compatible train/val transforms
    - sbi_dataset: SBI video dataset with pre-caching
"""

from data.splits import load_split, load_all_splits, get_video_ids
from data.augmentation import (
    get_train_transforms,
    get_val_transforms,
    apply_transform_to_clip,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from data.sbi_dataset import SBIVideoDataset

__all__ = [
    "load_split",
    "load_all_splits",
    "get_video_ids",
    "get_train_transforms",
    "get_val_transforms",
    "apply_transform_to_clip",
    "SBIVideoDataset",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
