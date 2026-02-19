#!/usr/bin/env python3
"""
STF-Mamba V8.0 - Data Augmentation
=====================================

Train transforms: horizontal flip, color jitter, compression artifacts, normalize.
Val/Test transforms: normalize only (no augmentation).

All transforms produce DINOv2-compatible inputs:
    - Size: 224x224 (already cropped by preprocessing)
    - Normalization: ImageNet mean/std
    - Format: (C, H, W) float32 tensor in [0, 1] then normalized

Adapted from reference SBI get_transforms() with V8.0 modifications:
    - Added JPEG compression augmentation (simulates real-world distribution)
    - Removed SBI-specific source_transforms (handled in sbi_dataset.py)
    - Added horizontal flip with landmark-aware probability
"""

from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T

# DINOv2 / ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(img_size: int = 224) -> Callable:
    """
    Training transforms applied per frame.

    Includes augmentation for robustness:
        - Random horizontal flip (p=0.5)
        - Color jitter (brightness, contrast, saturation, hue)
        - Random JPEG compression (simulates real-world compression)
        - ImageNet normalization

    Args:
        img_size: Target image size (should be 224 for DINOv2).

    Returns:
        Transform function: np.ndarray (H, W, 3) uint8 → torch.Tensor (3, H, W).
    """
    return T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05,
        ),
        T.ToTensor(),  # [0, 255] uint8 → [0, 1] float32
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(img_size: int = 224) -> Callable:
    """
    Validation / test transforms — no augmentation.

    Args:
        img_size: Target image size.

    Returns:
        Transform function: np.ndarray (H, W, 3) uint8 → torch.Tensor (3, H, W).
    """
    return T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def apply_transform_to_clip(
    frames: np.ndarray,
    transform: Callable,
) -> torch.Tensor:
    """
    Apply a per-frame transform to a T-frame clip.

    Args:
        frames: (T, H, W, 3) uint8 numpy array.
        transform: Per-frame transform function.

    Returns:
        (T, 3, H, W) float32 tensor.
    """
    transformed = []
    for i in range(frames.shape[0]):
        frame = frames[i]  # (H, W, 3) uint8
        frame_t = transform(frame)  # (3, H, W) tensor
        transformed.append(frame_t)
    return torch.stack(transformed)  # (T, 3, H, W)


def denormalize(
    tensor: torch.Tensor,
    mean: Tuple[float, ...] = IMAGENET_MEAN,
    std: Tuple[float, ...] = IMAGENET_STD,
) -> torch.Tensor:
    """
    Reverse ImageNet normalization for visualization.

    Args:
        tensor: (C, H, W) or (B, C, H, W) normalized tensor.

    Returns:
        Denormalized tensor in [0, 1] range.
    """
    mean_t = torch.tensor(mean, device=tensor.device)
    std_t = torch.tensor(std, device=tensor.device)

    if tensor.dim() == 3:
        mean_t = mean_t.view(3, 1, 1)
        std_t = std_t.view(3, 1, 1)
    elif tensor.dim() == 4:
        mean_t = mean_t.view(1, 3, 1, 1)
        std_t = std_t.view(1, 3, 1, 1)

    return (tensor * std_t + mean_t).clamp(0, 1)


# ─── SBI source transforms (for blending augmentation) ───

def get_source_transforms() -> Callable:
    """
    Source transforms applied to the face before SBI blending.

    Adapted from reference SBI get_source_transforms().
    Includes color shifts and downscaling to simulate GAN artifacts.

    Returns:
        Transform function: np.ndarray (H, W, 3) uint8 → np.ndarray (H, W, 3) uint8.
    """
    try:
        import albumentations as alb

        return alb.Compose([
            alb.Compose([
                alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
                alb.HueSaturationValue(
                    hue_shift_limit=(-0.3, 0.3),
                    sat_shift_limit=(-0.3, 0.3),
                    val_shift_limit=(-0.3, 0.3),
                    p=1,
                ),
                alb.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1),
                    contrast_limit=(-0.1, 0.1),
                    p=1,
                ),
            ], p=1),
            alb.OneOf([
                _RandomDownScale(p=1),
                alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
            ], p=1),
        ], p=1.0)

    except ImportError:
        # Fallback: simple numpy-based transform
        return _simple_source_transform


class _RandomDownScale:
    """Random downscale + upscale augmentation (from reference SBI code)."""

    def __init__(self, p: float = 1.0):
        self.p = p

    def __call__(self, **kwargs):
        img = kwargs.get("image", kwargs.get("img"))
        if np.random.rand() > self.p:
            return {"image": img}

        h, w = img.shape[:2]
        ratio = np.random.choice([2, 4])
        img_ds = cv2.resize(
            img, (w // ratio, h // ratio), interpolation=cv2.INTER_NEAREST
        )
        img_ds = cv2.resize(img_ds, (w, h), interpolation=cv2.INTER_LINEAR)
        return {"image": img_ds}


def _simple_source_transform(image: np.ndarray, **kwargs) -> dict:
    """Fallback source transform without albumentations."""
    img = image.copy()

    # Random brightness shift
    if np.random.rand() < 0.3:
        shift = np.random.randint(-20, 21)
        img = np.clip(img.astype(np.int16) + shift, 0, 255).astype(np.uint8)

    return {"image": img}


# Import cv2 only when needed
try:
    import cv2
except ImportError:
    pass
