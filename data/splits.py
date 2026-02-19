"""
STF-Mamba V8.0 - Official FF++ Split Loader
==============================================

Loads Dataset_Split_train/val/test.json.
Each entry: [real_video_id_1, real_video_id_2] -- both are real FF++ videos.
Train: 360 pairs | Val: 70 pairs | Test: 70 pairs

CRITICAL: Video IDs in train NEVER appear in val or test.
Celeb-DF is TEST ONLY -- never used for training decisions.
"""

# TODO: Stage 3 implementation
