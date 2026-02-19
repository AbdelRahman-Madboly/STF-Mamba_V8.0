#!/usr/bin/env python3
"""
STF-Mamba V8.0 - Face Preprocessing
======================================

dlib 81-point landmark detection → face bounding box → 1.3x expand → 224x224 crop.

Caching strategy (CRITICAL — never load videos at __getitem__ time):
    - Face crops:  cache/{video_id}_crops.npz     → array of (T, H, W, 3)
    - Landmarks:   cache/{video_id}_landmarks.npz  → array of (T, 81, 2)

If cache exists → load instantly.
If not → extract from video, crop faces, save to cache.

Adapted from reference: SelfBlendedImages/src/preprocess/crop_dlib_ff.py
"""

import logging
import os
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Lazy-load dlib (not always available)
_face_detector = None
_face_predictor = None


def _get_dlib_models(predictor_path: Optional[str] = None):
    """Lazy-load dlib face detector and 81-point landmark predictor."""
    global _face_detector, _face_predictor

    if _face_detector is not None:
        return _face_detector, _face_predictor

    import dlib
    from imutils import face_utils

    _face_detector = dlib.get_frontal_face_detector()

    if predictor_path is None:
        # Common locations
        candidates = [
            "shape_predictor_81_face_landmarks.dat",
            "src/preprocess/shape_predictor_81_face_landmarks.dat",
            os.path.expanduser("~/.dlib/shape_predictor_81_face_landmarks.dat"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                predictor_path = c
                break

    if predictor_path is None or not os.path.isfile(predictor_path):
        raise FileNotFoundError(
            "dlib shape_predictor_81_face_landmarks.dat not found. "
            "Download from: https://github.com/codeniko/shape_predictor_81_face_landmarks"
        )

    _face_predictor = dlib.shape_predictor(predictor_path)
    logger.info(f"dlib models loaded (predictor: {predictor_path})")
    return _face_detector, _face_predictor


class FacePreprocessor:
    """
    Extract and cache face crops + landmarks from FF++ videos.

    Usage:
        preprocessor = FacePreprocessor(
            video_dir="/path/to/ff++/videos",
            cache_dir="cache/crops",
            num_frames=32,
            img_size=224,
        )
        # First call extracts and caches; subsequent calls load from cache
        crops, landmarks = preprocessor.get_video("071")
    """

    def __init__(
        self,
        video_dir: str,
        cache_dir: str,
        num_frames: int = 32,
        img_size: int = 224,
        crop_expand: float = 1.3,
        predictor_path: Optional[str] = None,
    ) -> None:
        self.video_dir = Path(video_dir)
        self.cache_dir = Path(cache_dir)
        self.num_frames = num_frames
        self.img_size = img_size
        self.crop_expand = crop_expand
        self.predictor_path = predictor_path

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_video(
        self, video_id: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get face crops and landmarks for a video, using cache.

        Args:
            video_id: FF++ video ID (e.g., "071").

        Returns:
            crops: (T, img_size, img_size, 3) uint8 face crops.
            landmarks: (T, 81, 2) float32 landmark coordinates.
        """
        crop_path = self.cache_dir / f"{video_id}_crops.npz"
        land_path = self.cache_dir / f"{video_id}_landmarks.npz"

        # Load from cache if available
        if crop_path.exists() and land_path.exists():
            crops = np.load(crop_path)["crops"]
            landmarks = np.load(land_path)["landmarks"]
            return crops, landmarks

        # Extract from video
        crops, landmarks = self._extract_from_video(video_id)

        # Save to cache
        np.savez_compressed(crop_path, crops=crops)
        np.savez_compressed(land_path, landmarks=landmarks)
        logger.debug(f"Cached {video_id}: {crops.shape[0]} frames")

        return crops, landmarks

    def _extract_from_video(
        self, video_id: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract face crops from a video file using dlib."""
        detector, predictor = _get_dlib_models(self.predictor_path)
        from imutils import face_utils

        # Find video file
        video_path = self._find_video(video_id)
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frame indices uniformly
        frame_idxs = np.linspace(
            0, total_frames - 1, self.num_frames, endpoint=True, dtype=int
        )

        crops_list = []
        landmarks_list = []

        for cnt in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            if cnt not in frame_idxs:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            faces = detector(frame_rgb, 1)
            if len(faces) == 0:
                # No face found — use center crop as fallback
                h, w = frame_rgb.shape[:2]
                crop = self._center_crop(frame_rgb)
                landmark = np.zeros((81, 2), dtype=np.float32)
            else:
                # Pick largest face
                face = max(faces, key=lambda f: (f.right() - f.left()) * (f.bottom() - f.top()))

                # Get landmarks
                shape = predictor(frame_rgb, face)
                landmark = face_utils.shape_to_np(shape).astype(np.float32)

                # Crop face with expanded bbox
                crop = self._crop_face(frame_rgb, landmark)

            crops_list.append(crop)
            landmarks_list.append(landmark)

        cap.release()

        # Pad if we got fewer frames than expected
        while len(crops_list) < self.num_frames:
            crops_list.append(crops_list[-1] if crops_list else np.zeros(
                (self.img_size, self.img_size, 3), dtype=np.uint8
            ))
            landmarks_list.append(landmarks_list[-1] if landmarks_list else np.zeros(
                (81, 2), dtype=np.float32
            ))

        crops = np.stack(crops_list[:self.num_frames])         # (T, H, W, 3)
        landmarks = np.stack(landmarks_list[:self.num_frames])  # (T, 81, 2)

        return crops, landmarks

    def _crop_face(
        self, img: np.ndarray, landmark: np.ndarray
    ) -> np.ndarray:
        """Crop face region using landmark bbox with 1.3x expansion."""
        h, w = img.shape[:2]

        x0 = landmark[:, 0].min()
        y0 = landmark[:, 1].min()
        x1 = landmark[:, 0].max()
        y1 = landmark[:, 1].max()

        face_w = x1 - x0
        face_h = y1 - y0
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2

        # Expand by crop_expand factor
        half_size = max(face_w, face_h) * self.crop_expand / 2

        x0_new = max(0, int(cx - half_size))
        y0_new = max(0, int(cy - half_size))
        x1_new = min(w, int(cx + half_size))
        y1_new = min(h, int(cy + half_size))

        crop = img[y0_new:y1_new, x0_new:x1_new]
        crop = cv2.resize(crop, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        return crop.astype(np.uint8)

    def _center_crop(self, img: np.ndarray) -> np.ndarray:
        """Fallback: center crop when no face is detected."""
        h, w = img.shape[:2]
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        crop = img[y0:y0 + s, x0:x0 + s]
        crop = cv2.resize(crop, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        return crop.astype(np.uint8)

    def _find_video(self, video_id: str) -> Path:
        """Locate video file by ID."""
        # Try common FF++ path patterns
        patterns = [
            f"{video_id}.mp4",
            f"{video_id}*.mp4",
        ]
        for pattern in patterns:
            matches = list(self.video_dir.glob(f"**/{pattern}"))
            if matches:
                return matches[0]

        raise FileNotFoundError(
            f"Video {video_id} not found in {self.video_dir}"
        )

    def preprocess_all(
        self, video_ids: List[str], show_progress: bool = True
    ) -> None:
        """Pre-cache all videos. Run once before training."""
        from tqdm import tqdm

        iterator = tqdm(video_ids, desc="Preprocessing") if show_progress else video_ids
        cached, extracted = 0, 0

        for vid in iterator:
            crop_path = self.cache_dir / f"{vid}_crops.npz"
            if crop_path.exists():
                cached += 1
                continue
            try:
                self.get_video(vid)
                extracted += 1
            except Exception as e:
                logger.warning(f"Failed to preprocess {vid}: {e}")

        logger.info(
            f"Preprocessing done: {cached} cached, {extracted} extracted"
        )
