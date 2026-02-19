#!/usr/bin/env python3
"""
STF-Mamba V8.0 - Official FF++ Split Loader
==============================================

Loads Dataset_Split_train/val/test.json.
Each entry: [real_video_id_1, real_video_id_2] — both are real FF++ videos.
Train: 360 pairs | Val: 70 pairs | Test: 70 pairs

CRITICAL: Video IDs in train NEVER appear in val or test (verified).
Celeb-DF is TEST ONLY — never used for training decisions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def load_split(
    split_path: str,
) -> List[Tuple[str, str]]:
    """
    Load an official SBI FF++ split file.

    Args:
        split_path: Path to Dataset_Split_{train,val,test}.json.

    Returns:
        List of (video_id_1, video_id_2) pairs.
        Both IDs are real FF++ videos used for SBI generation.
    """
    with open(split_path, "r") as f:
        pairs = json.load(f)

    logger.info(f"Loaded split: {Path(split_path).name} — {len(pairs)} pairs")
    return [(p[0], p[1]) for p in pairs]


def load_all_splits(
    splits_dir: str,
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Load all three official splits.

    Args:
        splits_dir: Directory containing the three JSON files.

    Returns:
        Dict with keys 'train', 'val', 'test', each a list of pairs.
    """
    splits_dir = Path(splits_dir)
    splits = {}
    for phase in ["train", "val", "test"]:
        path = splits_dir / f"Dataset_Split_{phase}.json"
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")
        splits[phase] = load_split(str(path))

    # Verify no ID overlap (should always pass with official splits)
    train_ids = {vid for pair in splits["train"] for vid in pair}
    val_ids = {vid for pair in splits["val"] for vid in pair}
    test_ids = {vid for pair in splits["test"] for vid in pair}

    overlap_tv = train_ids & val_ids
    overlap_tt = train_ids & test_ids
    overlap_vt = val_ids & test_ids

    if overlap_tv or overlap_tt or overlap_vt:
        logger.error(
            f"DATA LEAK DETECTED! "
            f"train∩val={len(overlap_tv)}, "
            f"train∩test={len(overlap_tt)}, "
            f"val∩test={len(overlap_vt)}"
        )
        raise ValueError("Split integrity violation — video ID overlap detected.")

    logger.info(
        f"Splits loaded: train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])} | "
        f"No ID overlap ✓"
    )
    return splits


def get_video_ids(pairs: List[Tuple[str, str]]) -> List[str]:
    """Extract unique video IDs from a list of pairs."""
    ids = set()
    for a, b in pairs:
        ids.add(a)
        ids.add(b)
    return sorted(ids)
