#!/usr/bin/env python3
"""
STF-Mamba V8.0 - SBI (Self-Blended Images) Video Dataset
===========================================================

Each sample is a T-frame video clip (real or SBI-fake).

Pipeline:
    1. Load official split → list of (vid_A, vid_B) pairs
    2. For each pair, produce 2 clips:
        - REAL clip: T face-cropped frames from vid_A
        - FAKE clip: SBI blend — vid_A's face structure + vid_B's face appearance
    3. SBI blending uses Poisson-style alpha blending with convex hull mask
    4. All SBI fakes are PRE-CACHED to disk (NPZ files)

WHY PRE-CACHE:
    On-the-fly SBI without caching was proven harmful in V7.3:
    360 source videos → different dataset every epoch → contradictory gradients.
    Pre-caching ensures consistent training signal.

Cache regeneration:
    Every N epochs (default 30), regenerate with new random seeds for diversity.

Returns per sample:
    {'frames': (T, 3, H, W), 'label': 0 or 1, 'video_id': str}
"""

import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SBIVideoDataset(Dataset):
    """
    SBI Video Dataset for STF-Mamba V8.0.

    Each pair (vid_A, vid_B) produces:
        - Index 2*i:   REAL clip from vid_A
        - Index 2*i+1: FAKE clip from SBI(vid_A, vid_B)

    Args:
        split_path: Path to Dataset_Split_{phase}.json.
        video_dir: Path to FF++ video directory (containing .mp4 files).
        cache_dir: Path for face crop and SBI caches.
        phase: 'train', 'val', or 'test'.
        num_frames: Frames per clip. Default: 32.
        img_size: Face crop size. Default: 224.
        transform: Per-frame transform (from augmentation.py).
        sbi_seed: Random seed for SBI generation. Change for cache regen.
        predictor_path: Path to dlib shape predictor .dat file.
    """

    def __init__(
        self,
        split_path: str,
        video_dir: str,
        cache_dir: str,
        phase: str = "train",
        num_frames: int = 32,
        img_size: int = 224,
        transform: Optional[Callable] = None,
        sbi_seed: int = 42,
        predictor_path: Optional[str] = None,
    ) -> None:
        assert phase in ("train", "val", "test")
        self.phase = phase
        self.num_frames = num_frames
        self.img_size = img_size
        self.transform = transform
        self.sbi_seed = sbi_seed

        # Load split pairs
        from data.splits import load_split
        self.pairs = load_split(split_path)

        # Setup preprocessing
        from data.preprocessing import FacePreprocessor
        self.preprocessor = FacePreprocessor(
            video_dir=video_dir,
            cache_dir=os.path.join(cache_dir, "crops"),
            num_frames=num_frames,
            img_size=img_size,
            predictor_path=predictor_path,
        )

        # SBI cache directory
        self.sbi_cache_dir = Path(cache_dir) / f"sbi_seed{sbi_seed}"
        self.sbi_cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"SBIVideoDataset({phase}): {len(self.pairs)} pairs → "
            f"{len(self)} clips (seed={sbi_seed})"
        )

    def __len__(self) -> int:
        """Each pair produces 2 clips: 1 real + 1 fake."""
        return len(self.pairs) * 2

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a clip.

        Even indices → real clip from vid_A.
        Odd indices → fake clip (SBI blend of vid_A + vid_B).

        Returns:
            dict with:
                'frames': (T, 3, H, W) float32 tensor.
                'label': int, 0=real, 1=fake.
                'video_id': str, source video ID.
        """
        pair_idx = idx // 2
        is_fake = (idx % 2 == 1)
        vid_a, vid_b = self.pairs[pair_idx]

        if is_fake:
            frames = self._get_sbi_frames(vid_a, vid_b)
            label = 1
            video_id = f"sbi_{vid_a}_{vid_b}"
        else:
            frames = self._get_real_frames(vid_a)
            label = 0
            video_id = vid_a

        # Apply transform
        if self.transform is not None:
            from data.augmentation import apply_transform_to_clip
            frames_t = apply_transform_to_clip(frames, self.transform)
        else:
            # Default: just convert to tensor (C, H, W) float [0, 1]
            frames_t = torch.from_numpy(
                frames.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            )

        return {
            "frames": frames_t,       # (T, 3, H, W)
            "label": label,            # 0 or 1
            "video_id": video_id,      # str
        }

    def _get_real_frames(self, video_id: str) -> np.ndarray:
        """Load real face crops from cache. Shape: (T, H, W, 3) uint8."""
        crops, _ = self.preprocessor.get_video(video_id)
        return crops[:self.num_frames]

    def _get_sbi_frames(self, vid_a: str, vid_b: str) -> np.ndarray:
        """
        Get SBI fake frames, loading from cache or generating.

        SBI: blend vid_B's face appearance onto vid_A's face structure.
        """
        cache_path = self.sbi_cache_dir / f"{vid_a}_{vid_b}.npz"

        if cache_path.exists():
            return np.load(cache_path)["frames"]

        # Generate SBI
        crops_a, landmarks_a = self.preprocessor.get_video(vid_a)
        crops_b, _ = self.preprocessor.get_video(vid_b)

        rng = np.random.RandomState(
            self.sbi_seed + hash(f"{vid_a}_{vid_b}") % (2**31)
        )

        sbi_frames = []
        for t in range(min(self.num_frames, len(crops_a), len(crops_b))):
            source = crops_b[t]  # Face appearance from vid_B
            target = crops_a[t]  # Face structure from vid_A
            landmark = landmarks_a[t]

            blended = self._self_blend(source, target, landmark, rng)
            sbi_frames.append(blended)

        # Pad if needed
        while len(sbi_frames) < self.num_frames:
            sbi_frames.append(sbi_frames[-1] if sbi_frames else np.zeros(
                (self.img_size, self.img_size, 3), dtype=np.uint8
            ))

        frames = np.stack(sbi_frames[:self.num_frames])

        # Cache
        np.savez_compressed(cache_path, frames=frames)
        return frames

    def _self_blend(
        self,
        source: np.ndarray,
        target: np.ndarray,
        landmark: np.ndarray,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """
        SBI blending: blend source face onto target using landmark-based mask.

        Adapted from reference SBI self_blending() method.

        Args:
            source: (H, W, 3) uint8 — face from vid_B (appearance donor).
            target: (H, W, 3) uint8 — face from vid_A (structure host).
            landmark: (81, 2) or (68, 2) landmarks on target face.
            rng: Seeded random state for reproducibility.

        Returns:
            (H, W, 3) uint8 blended face.
        """
        h, w = target.shape[:2]

        # Create convex hull mask from landmarks
        mask = self._create_face_mask(landmark, h, w)

        # Apply random affine + elastic to source (simulates GAN misalignment)
        source_warped, mask_warped = self._random_warp(source, mask, rng)

        # Optionally apply source color transforms
        if rng.rand() < 0.5:
            source_warped = self._color_jitter(source_warped, rng)

        # Dynamic alpha blending
        blend_ratio = rng.choice([0.25, 0.5, 0.75, 1.0, 1.0, 1.0])
        mask_blurred = self._blur_mask(mask_warped, rng) * blend_ratio

        # Ensure shapes match
        if source_warped.shape != target.shape:
            source_warped = cv2.resize(source_warped, (w, h))
        if mask_blurred.shape[:2] != (h, w):
            mask_blurred = cv2.resize(mask_blurred, (w, h))

        # Ensure mask has channel dim
        if mask_blurred.ndim == 2:
            mask_blurred = mask_blurred[:, :, np.newaxis]

        # Blend
        blended = (
            mask_blurred * source_warped.astype(np.float32)
            + (1 - mask_blurred) * target.astype(np.float32)
        )
        return blended.clip(0, 255).astype(np.uint8)

    def _create_face_mask(
        self, landmark: np.ndarray, h: int, w: int
    ) -> np.ndarray:
        """Create binary face mask from landmarks using convex hull."""
        mask = np.zeros((h, w), dtype=np.float32)

        # Use first 68 landmarks (face outline) if available
        pts = landmark[:min(68, len(landmark))].astype(np.int32)

        # Filter out zero landmarks (no detection)
        valid = (pts[:, 0] > 0) | (pts[:, 1] > 0)
        if valid.sum() < 3:
            # Not enough landmarks — use center ellipse as fallback
            cv2.ellipse(
                mask, (w // 2, h // 2), (w // 3, h // 3),
                0, 0, 360, 1.0, -1
            )
            return mask

        pts = pts[valid]
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 1.0)
        return mask

    def _blur_mask(
        self, mask: np.ndarray, rng: np.random.RandomState
    ) -> np.ndarray:
        """Blur the blend mask for smooth transitions."""
        # Two-pass Gaussian blur (from reference SBI)
        k1 = rng.randint(3, 13) * 2 + 1  # Odd kernel size
        k2 = rng.randint(3, 13) * 2 + 1

        blurred = cv2.GaussianBlur(mask, (k1, k1), 0)
        if blurred.max() > 0:
            blurred = blurred / blurred.max()
            blurred[blurred < 1] = 0

        blurred = cv2.GaussianBlur(blurred, (k2, k2), rng.randint(5, 46))
        if blurred.max() > 0:
            blurred = blurred / blurred.max()

        return blurred

    def _random_warp(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random affine warp to source + mask (simulates GAN misalignment)."""
        h, w = img.shape[:2]

        # Small random translation + scale
        tx = rng.uniform(-0.03, 0.03) * w
        ty = rng.uniform(-0.015, 0.015) * h
        scale = rng.uniform(0.95, 1.05)

        M = np.array([
            [scale, 0, tx],
            [0, scale, ty],
        ], dtype=np.float32)

        img_warped = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        mask_warped = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT)

        return img_warped, mask_warped

    def _color_jitter(
        self, img: np.ndarray, rng: np.random.RandomState
    ) -> np.ndarray:
        """Simple color jitter for the source face."""
        img = img.astype(np.int16)
        shift = rng.randint(-15, 16, size=3)
        img = img + shift[np.newaxis, np.newaxis, :]
        return img.clip(0, 255).astype(np.uint8)

    def build_cache(self, show_progress: bool = True) -> None:
        """
        Pre-build all SBI caches. Run once before training.

        This ensures every __getitem__ call loads from disk.
        """
        from tqdm import tqdm

        iterator = tqdm(
            self.pairs, desc=f"Building SBI cache ({self.phase})"
        ) if show_progress else self.pairs

        cached, generated = 0, 0
        for vid_a, vid_b in iterator:
            cache_path = self.sbi_cache_dir / f"{vid_a}_{vid_b}.npz"
            if cache_path.exists():
                cached += 1
                continue
            try:
                self._get_sbi_frames(vid_a, vid_b)
                generated += 1
            except Exception as e:
                logger.warning(f"SBI generation failed for {vid_a}+{vid_b}: {e}")

        logger.info(
            f"SBI cache ({self.phase}): {cached} cached, {generated} generated"
        )

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate for DataLoader."""
        frames = torch.stack([b["frames"] for b in batch])  # (B, T, 3, H, W)
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        video_ids = [b["video_id"] for b in batch]

        return {
            "frames": frames,
            "label": labels,
            "video_id": video_ids,
        }

    def worker_init_fn(self, worker_id: int) -> None:
        """Ensure different random state per worker (Kaggle compatibility)."""
        np.random.seed(np.random.get_state()[1][0] + worker_id)
