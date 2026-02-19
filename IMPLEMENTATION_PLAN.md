# STF-Mamba V8.0 — Complete Cursor Implementation Plan

> **Author:** Abdel Rahman Madboly  
> **Date:** February 2026  
> **Repo:** https://github.com/AbdelRahman-Madboly/STF-Mamba_V8.0.git  
> **Goal:** Fix the 51% → 95%+ AUC gap on real deepfakes

---

## CRITICAL: Save Your Data First

Your preprocessed FF++ data (frames/ + landmarks/ + retina/) is on **ephemeral pod storage**. It will be permanently deleted when you terminate the pod. Before doing anything else:

```bash
# Option A: Create a RunPod Network Volume (RECOMMENDED)
# 1. In RunPod dashboard → Storage → Create Network Volume → 50GB → same region as pod
# 2. Then on your running pod:
mkdir -p /runpod-volume/data
cp -r /workspace/data/preprocessed /runpod-volume/data/
cp -r /workspace/data/FaceForensics++ /runpod-volume/data/  # if you have raw videos too
# Verify
du -sh /runpod-volume/data/*

# Option B: Tar and download to local machine
cd /workspace/data
tar czf /workspace/ff_preprocessed.tar.gz preprocessed/
# Then download via RunPod's file manager or SCP
```

---

## Project Structure (New, Clean)

```
STF-Mamba_V8.0/
├── configs/
│   ├── v8_a100.yaml              # A100 80GB config
│   └── v8_4090.yaml              # RTX 4090 24GB config
├── src/
│   ├── __init__.py
│   ├── models/                   # All model code
│   │   ├── __init__.py
│   │   ├── config.py             # STFV8Config with feature flags
│   │   ├── backbone_v73.py       # V7.3 backbone (ported, working)
│   │   ├── backbone_v8.py        # V8.0 backbone (Hydra + Mamer + W-SS)
│   │   ├── dwt_3d.py             # 3D-DWT module (from V7.3)
│   │   ├── dcconv_3d.py          # 3D Dynamic Contour Conv (from V7.3)
│   │   ├── mamba_blocks.py       # V7.3 Bi-Mamba blocks
│   │   ├── transformer_blocks.py # V7.3 Transformer blocks
│   │   ├── hydra_mixer.py        # NEW: Quasiseparable Hydra + Mamer
│   │   ├── wss_module.py         # NEW: Wavelet-Selective Scan
│   │   ├── losses.py             # NEW: Contrastive HLL + full loss
│   │   ├── drop_path.py          # DropPath utility
│   │   └── wavelet_constants.py  # Symlet-2 filter banks
│   ├── data/                     # Data pipeline
│   │   ├── __init__.py
│   │   ├── video_sbi_dataset.py  # Video-SBI dataset (V8 augmentations)
│   │   ├── augmentations.py      # Compression aug, multi-scale drift, TPS
│   │   ├── blend.py              # SBI blending (from reference)
│   │   └── funcs.py              # crop_face, IoU, etc. (from reference)
│   ├── training/                 # Training pipeline
│   │   ├── __init__.py
│   │   ├── train.py              # Main training script
│   │   ├── evaluate.py           # Cross-dataset evaluation
│   │   └── budget_tracker.py     # Cost monitoring
│   └── utils/                    # Shared utilities
│       ├── __init__.py
│       ├── logging.py            # CSV + console logging
│       └── checkpoint.py         # Save/load with flexible keys
├── scripts/
│   ├── setup_pod.sh              # RunPod env setup (installs deps, mounts data)
│   ├── preprocess.py             # Face preprocessing (if needed from scratch)
│   └── verify_data.py            # Data integrity check
├── splits/
│   ├── train.json                # FF++ train split
│   ├── val.json                  # FF++ val split
│   └── test.json                 # FF++ test split
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Stage Overview

| # | Stage | Where | Time | Validation Check |
|---|-------|-------|------|------------------|
| 0 | Project Scaffold | Local/Cursor | 15 min | `python -c "from src.models.config import STFV8Config"` |
| 1 | Data Pipeline | Local/Cursor | 1.5 hrs | Unit test: dataset returns correct tensor shapes |
| 2 | V7.3 Backbone Port | Local/Cursor | 1 hr | `model(dummy_input)` produces (B,2) logits |
| 3 | V8.0 Losses + Augmentation | Local/Cursor | 1 hr | Loss backward() works, augmentation visual check |
| 4 | V8.0 Hydra + Mamer | Local/Cursor | 1.5 hrs | Mixer forward pass with correct shapes |
| 5 | V8.0 W-SS + Full Backbone | Local/Cursor | 1.5 hrs | Full V8 model forward pass, parameter count check |
| 6 | Training Script | Local/Cursor | 1.5 hrs | Dry run: 2 epochs on dummy data |
| 7 | RunPod Setup + Data Mount | RunPod | 20 min | `verify_data.py` passes all checks |
| 8 | Phase A Training | RunPod | 2-6 hrs | AUC > 70% on real FF++ deepfakes |
| 9 | Evaluation + Phase B | RunPod | 1-3 hrs | Cross-dataset AUC improvement |

**Total Cursor time:** ~8 hours of coding (no GPU needed)  
**Total GPU time:** ~3-9 hours depending on GPU choice

---

## STAGE 0: Project Scaffold

**Goal:** Create the project skeleton, push to GitHub, verify imports work.

### Step 0.1: Initialize Git repo

```bash
cd C:\Dan_WS\STF-Mamba_V8.0
# If starting fresh:
git init
git remote add origin https://github.com/AbdelRahman-Madboly/STF-Mamba_V8.0.git
```

### Step 0.2: Create .gitignore

```gitignore
# Model checkpoints
*.pth
*.pt
*.tar
outputs/checkpoints/

# Data (too large for git)
data/
*.mp4
*.npz
*.npy

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.eggs/
dist/
build/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# RunPod
wandb/
*.log
runpod-volume/

# Large files
*.zip
*.tar.gz
```

### Step 0.3: Create requirements.txt

```txt
torch>=2.1.0
torchvision>=0.16.0
timm>=0.9.12
numpy>=1.24.0
opencv-python>=4.8.0
albumentations>=1.3.1
scikit-learn>=1.3.0
scipy>=1.11.0
Pillow>=10.0.0
PyYAML>=6.0
tqdm>=4.66.0
pandas>=2.0.0
matplotlib>=3.7.0
imutils>=0.5.4
dlib>=19.24.0
```

### Step 0.4: Create src/models/config.py

This is the foundation — the feature flag system that controls V7.3 vs V8.0:

```python
"""STF-Mamba V8.0 Configuration with Feature Flags."""
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class STFV8Config:
    # === INPUT ===
    in_channels: int = 3
    num_classes: int = 2
    num_frames: int = 32
    img_size: int = 224  # CHANGED from 256 to match ConvNeXt V2 pretraining

    # === BACKBONE (V7.3 base) ===
    stem_dim: int = 48
    stage1_dim: int = 96
    stage2_dim: int = 192
    stage3_dim: int = 384
    stage4_dim: int = 384

    # === BLOCK COUNTS ===
    stage1_blocks: int = 2
    stage2_blocks: int = 2
    stage3_blocks: int = 4
    stage4_blocks: int = 2

    # === DWT ===
    dwt_channels: int = 64
    dwt_hidden_dim: int = 64

    # === SSM ===
    d_state: int = 16
    d_conv: int = 4
    ssm_expand: int = 2

    # === ATTENTION ===
    num_heads: int = 8
    mlp_ratio: float = 2.0

    # === REGULARIZATION ===
    dropout: float = 0.3
    drop_path: float = 0.1

    # === 3D-DCConv ===
    dcconv_kernel_length: int = 9

    # ============================================
    # V8.0 FEATURE FLAGS (the upgrade system)
    # ============================================

    # Phase A fixes (loss + augmentation, same architecture)
    use_contrastive_hll: bool = False    # Contrastive HLL loss vs MSE(HLL,0)
    use_compression_aug: bool = False    # Random H.264 re-compression
    use_multiscale_drift: bool = False   # Multi-scale Brownian drift (0.5-5px)
    use_hll_normalization: bool = False  # Normalize HLL by total band energy

    # Phase B architecture upgrades
    use_hydra: bool = False              # Quasiseparable Hydra Mixer
    use_mamer: bool = False              # Mamba-Transformer hybrid blocks
    use_wss: bool = False                # Wavelet-Selective Scan
    use_adaptive_sdim: bool = False      # Dynamic state dimension

    # Training
    lr: float = 5e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    label_smoothing: float = 0.1
    hll_loss_weight: float = 0.1
    hll_loss_margin: float = 1.0
    max_grad_norm: float = 1.0
    grad_accum: int = 2
    amp_dtype: str = "bfloat16"

    # Budget
    cost_per_hour: float = 1.39
    max_budget: float = 5.0

    @classmethod
    def phase_a(cls, gpu="a100"):
        """V7.3 architecture + all critical fixes."""
        cfg = cls(
            use_contrastive_hll=True,
            use_compression_aug=True,
            use_multiscale_drift=True,
            use_hll_normalization=True,
        )
        if "4090" in gpu.lower():
            cfg.cost_per_hour = 0.39
            cfg.grad_accum = 4  # smaller batch, more accumulation
        return cfg

    @classmethod
    def phase_b(cls, gpu="a100"):
        """Full V8.0 with all features."""
        cfg = cls.phase_a(gpu)
        cfg.use_hydra = True
        cfg.use_mamer = True
        cfg.use_wss = True
        cfg.use_adaptive_sdim = True
        cfg.lr = 1e-4  # Lower LR for fine-tuning
        return cfg

    @classmethod
    def debug(cls):
        """Minimal config for CPU testing."""
        return cls(
            num_frames=4, img_size=64,
            stem_dim=16, stage1_dim=32, stage2_dim=64,
            stage3_dim=128, stage4_dim=128,
            stage1_blocks=1, stage2_blocks=1,
            stage3_blocks=1, stage4_blocks=1,
            d_state=4, dwt_channels=16, dwt_hidden_dim=16,
        )

    def describe(self):
        flags = []
        if self.use_contrastive_hll: flags.append("ContrastiveHLL")
        if self.use_compression_aug: flags.append("CompressionAug")
        if self.use_multiscale_drift: flags.append("MultiScaleDrift")
        if self.use_hll_normalization: flags.append("HLLNorm")
        if self.use_hydra: flags.append("Hydra")
        if self.use_mamer: flags.append("Mamer")
        if self.use_wss: flags.append("W-SS")
        if self.use_adaptive_sdim: flags.append("AdaptiveSDim")
        mode = "Phase B (V8.0)" if self.use_hydra else "Phase A (V7.3+fixes)" if self.use_contrastive_hll else "Baseline V7.3"
        return f"{mode}: [{', '.join(flags) or 'none'}]"
```

### Step 0.5: Create __init__.py files

```python
# src/__init__.py
# src/models/__init__.py
# src/data/__init__.py
# src/training/__init__.py
# src/utils/__init__.py
# All empty for now
```

### Step 0.6: Validation Check

```bash
cd C:\Dan_WS\STF-Mamba_V8.0
python -c "from src.models.config import STFV8Config; c = STFV8Config.phase_a(); print(c.describe())"
# Expected: "Phase A (V7.3+fixes): [ContrastiveHLL, CompressionAug, MultiScaleDrift, HLLNorm]"
```

### Step 0.7: Initial Git push

```bash
git add .
git commit -m "Stage 0: project scaffold with feature flag config system"
git push -u origin main
```

---

## STAGE 1: Data Pipeline

**Goal:** Build the Video-SBI dataset class that works with your preprocessed data format AND includes V8.0 augmentations.

**Key difference from V7.3:** The dataset class now supports feature-flag-controlled augmentations (compression, multi-scale drift) without changing the underlying SBI generation logic.

### Step 1.1: Port blend.py from SBI reference

Copy these functions exactly as-is from the SBI reference (they work correctly):
- `alpha_blend(source, target, mask)`
- `dynamic_blend(source, target, mask)`
- `get_blend_mask(mask)`

File: `src/data/blend.py`

### Step 1.2: Port funcs.py from SBI reference

Copy these functions:
- `crop_face(img, landmark, bbox, ...)`
- `IoUfrom2bboxes(boxA, boxB)`
- `RandomDownScale` class

File: `src/data/funcs.py`

### Step 1.3: Create augmentations.py (NEW for V8.0)

This is new code. Contains the three V8 augmentation fixes:

```python
"""V8.0 Augmentation Upgrades — compression, multi-scale drift, TPS deformation."""
import numpy as np
import cv2
import subprocess
import tempfile
import os
import random
import albumentations as alb


class CompressionAugmentation:
    """Random H.264 re-compression to build codec invariance.
    
    This is Fix #1 for the HLL frequency bias problem.
    By training on videos with varying compression levels,
    the model learns HLL patterns that survive re-compression.
    """
    def __init__(self, qp_range=(23, 40), probability=0.5):
        self.qp_min, self.qp_max = qp_range
        self.probability = probability
    
    def __call__(self, frames):
        """Apply random H.264 compression to a list of frames.
        
        Args:
            frames: list of np.ndarray (H, W, 3) uint8 BGR frames
        Returns:
            list of compressed frames
        """
        if random.random() > self.probability:
            return frames
        
        qp = random.randint(self.qp_min, self.qp_max)
        
        try:
            return self._compress_ffmpeg(frames, qp)
        except Exception:
            # Fallback: JPEG compression per-frame (less realistic but works)
            quality = max(20, 100 - qp * 2)  # Map QP to JPEG quality
            compressed = []
            for f in frames:
                _, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, quality])
                compressed.append(cv2.imdecode(buf, cv2.IMREAD_COLOR))
            return compressed
    
    def _compress_ffmpeg(self, frames, qp):
        """Use ffmpeg for real H.264 compression."""
        h, w = frames[0].shape[:2]
        n = len(frames)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = os.path.join(tmpdir, 'input.raw')
            out_path = os.path.join(tmpdir, 'output.mp4')
            dec_path = os.path.join(tmpdir, 'decoded.raw')
            
            # Write raw frames
            raw = np.stack(frames).tobytes()
            with open(raw_path, 'wb') as f:
                f.write(raw)
            
            # Encode with H.264
            subprocess.run([
                'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                '-s', f'{w}x{h}', '-r', '30', '-i', raw_path,
                '-c:v', 'libx264', '-qp', str(qp), '-preset', 'ultrafast',
                out_path
            ], capture_output=True, timeout=30)
            
            # Decode back
            subprocess.run([
                'ffmpeg', '-y', '-i', out_path,
                '-f', 'rawvideo', '-pix_fmt', 'bgr24', dec_path
            ], capture_output=True, timeout=30)
            
            # Read decoded frames
            with open(dec_path, 'rb') as f:
                decoded_raw = f.read()
            
            decoded = np.frombuffer(decoded_raw, dtype=np.uint8)
            frame_size = h * w * 3
            n_decoded = len(decoded) // frame_size
            
            result = []
            for i in range(min(n, n_decoded)):
                frame = decoded[i*frame_size:(i+1)*frame_size].reshape(h, w, 3)
                result.append(frame.copy())
            
            # Pad if some frames were lost
            while len(result) < n:
                result.append(result[-1].copy())
            
            return result


class MultiScaleBrownianDrift:
    """Multi-scale Brownian mask drift (replaces fixed ±2px).
    
    This is Fix #2 for augmentation overfitting.
    Uses log-normal distribution for step sizes, with varying
    velocity profiles to match diverse real-world deepfake dynamics.
    """
    def __init__(self, step_range=(0.5, 5.0), max_cumulative=8.0, probability=0.8):
        self.step_min, self.step_max = step_range
        self.max_cumulative = max_cumulative
        self.probability = probability
    
    def apply_drift(self, base_mask, n_frames):
        """Apply multi-scale Brownian drift to a mask sequence.
        
        Args:
            base_mask: np.ndarray (H, W) float32 in [0, 1]
            n_frames: int, number of frames
        Returns:
            list of n_frames masks with drifted positions
        """
        if random.random() > self.probability:
            return [base_mask.copy() for _ in range(n_frames)]
        
        # Sample step size from log-normal (wider distribution than uniform)
        log_min, log_max = np.log(self.step_min), np.log(self.step_max)
        step_size = np.exp(random.uniform(log_min, log_max))
        
        # Choose drift profile
        profile = random.choice(['smooth', 'jittery', 'accelerating', 'oscillating'])
        
        # Generate cumulative drift
        dx_cum, dy_cum = 0.0, 0.0
        masks = [base_mask.copy()]
        
        for t in range(1, n_frames):
            if profile == 'smooth':
                dx = np.random.normal(0, step_size * 0.5)
                dy = np.random.normal(0, step_size * 0.5)
            elif profile == 'jittery':
                dx = np.random.uniform(-step_size, step_size)
                dy = np.random.uniform(-step_size, step_size)
            elif profile == 'accelerating':
                scale = step_size * (t / n_frames)
                dx = np.random.normal(0, max(0.1, scale))
                dy = np.random.normal(0, max(0.1, scale))
            elif profile == 'oscillating':
                freq = random.uniform(0.5, 3.0)
                dx = step_size * np.sin(2 * np.pi * freq * t / n_frames) + np.random.normal(0, step_size * 0.2)
                dy = step_size * np.cos(2 * np.pi * freq * t / n_frames) + np.random.normal(0, step_size * 0.2)
            
            dx_cum += dx
            dy_cum += dy
            
            # Clamp cumulative drift
            mag = np.sqrt(dx_cum**2 + dy_cum**2)
            if mag > self.max_cumulative:
                dx_cum *= self.max_cumulative / mag
                dy_cum *= self.max_cumulative / mag
            
            # Apply translation
            M = np.float32([[1, 0, dx_cum], [0, 1, dy_cum]])
            shifted = cv2.warpAffine(base_mask, M, (base_mask.shape[1], base_mask.shape[0]),
                                      borderMode=cv2.BORDER_REFLECT)
            masks.append(shifted)
        
        return masks


class TPSMaskDeformation:
    """Thin-Plate Spline non-rigid mask deformation.
    
    Optional upgrade: creates locally varying, non-rigid boundary
    motion instead of pure translational drift.
    """
    def __init__(self, n_control_points=8, max_displacement=3.0, probability=0.3):
        self.n_control = n_control_points
        self.max_disp = max_displacement
        self.probability = probability
    
    def __call__(self, mask):
        """Apply random TPS deformation to a mask."""
        if random.random() > self.probability:
            return mask
        
        h, w = mask.shape[:2]
        
        # Generate control points on a grid
        n = self.n_control
        src_pts = []
        for i in range(n):
            for j in range(n):
                x = w * (j + 0.5) / n
                y = h * (i + 0.5) / n
                src_pts.append([x, y])
        src_pts = np.array(src_pts, dtype=np.float32)
        
        # Random displacements
        dst_pts = src_pts.copy()
        dst_pts += np.random.uniform(-self.max_disp, self.max_disp, dst_pts.shape).astype(np.float32)
        
        # Compute TPS warp using OpenCV
        tps = cv2.createThinPlateSplineShapeTransformer()
        src_pts_cv = src_pts.reshape(1, -1, 2)
        dst_pts_cv = dst_pts.reshape(1, -1, 2)
        matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]
        tps.estimateTransformation(dst_pts_cv, src_pts_cv, matches)
        
        mask_3ch = cv2.cvtColor(mask if mask.ndim == 2 else mask, cv2.COLOR_GRAY2BGR) if mask.ndim == 2 else mask
        warped = tps.warpImage(mask_3ch)
        
        if mask.ndim == 2:
            warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        
        return warped
```

### Step 1.4: Create video_sbi_dataset.py

This is the MAIN dataset class. It adapts V7.3's `CachedSBIDataset` with V8 augmentations controlled by config flags:

**Key design decisions:**
- Reads from the SAME preprocessed data format (crops/*.npz, landmarks/*.npz)
- Feature flags control which augmentations are active
- Compression augmentation happens AFTER SBI blending (simulates real-world re-compression)
- Multi-scale drift replaces fixed ±2px

File: `src/data/video_sbi_dataset.py`

**Port from V7.3 `train_final.py`:** Lines 88-226 (the `CachedSBIDataset` class), then modify:
1. Add `CompressionAugmentation` call after `_make_sbi()` (if `config.use_compression_aug`)
2. Replace `BrownianMaskDrift(max_step=1.0, max_cumulative=2.0)` with `MultiScaleBrownianDrift()` (if `config.use_multiscale_drift`)
3. Accept `STFV8Config` instead of the old `Config` dataclass
4. Make data paths configurable (not hardcoded to `/workspace/`)

### Step 1.5: Validation Check

```python
# test_dataset.py (temporary, don't commit)
from src.models.config import STFV8Config
from src.data.video_sbi_dataset import VideoSBIDataset

cfg = STFV8Config.debug()
# Mock test: create dummy .npz files matching expected format
# Then verify:
# ds = VideoSBIDataset('train', cfg, data_root='/path/to/data')
# sample = ds[0]
# assert sample['video'].shape == (3, 4, 64, 64)  # (C, T, H, W) for debug config
# assert sample['label'] in [0, 1]
print("Stage 1 PASS: dataset returns correct shapes")
```

### Step 1.6: Git commit

```bash
git add src/data/
git commit -m "Stage 1: Video-SBI data pipeline with V8.0 augmentations"
```

---

## STAGE 2: V7.3 Backbone Port

**Goal:** Port the existing V7.3 model files into the new project structure. These MUST work before adding V8.0 components.

### Step 2.1: Copy V7.3 modules (direct port)

Copy these files from your existing `stf_mamba_backbone/` into `src/models/`:
- `dwt_3d.py` → `src/models/dwt_3d.py`
- `dcconv_3d.py` → `src/models/dcconv_3d.py`
- `mamba_blocks.py` → `src/models/mamba_blocks.py`
- `transformer_blocks.py` → `src/models/transformer_blocks.py`
- `drop_path.py` → `src/models/drop_path.py`
- `wavelet_constants.py` → `src/models/wavelet_constants.py`

**Fix imports:** Change `from .config import STFMambaConfig` to `from .config import STFV8Config` everywhere, since the V8 config is a superset of V7.3.

### Step 2.2: Create backbone_v73.py (renamed port)

Copy `backbone.py` → `src/models/backbone_v73.py`

**Changes:**
1. Import `STFV8Config` instead of `STFMambaConfig`
2. Class name stays `STFMambaV73`
3. The model output should be a dict: `{'logits': logits, 'hll_energy': hll_energy}`
   - `hll_energy` is needed for the contrastive HLL loss in Stage 3

### Step 2.3: Validation Check

```python
# test_backbone.py
import torch
from src.models.config import STFV8Config
from src.models.backbone_v73 import STFMambaV73

cfg = STFV8Config.debug()
model = STFMambaV73(cfg)
x = torch.randn(2, 3, 4, 64, 64)  # (B, C, T, H, W) debug size
out = model(x)
print(f"Logits shape: {out['logits'].shape}")  # Should be (2, 2)
print(f"HLL energy shape: {out['hll_energy'].shape}")  # Should be (2,) or similar
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
print("Stage 2 PASS")
```

### Step 2.4: Git commit

```bash
git add src/models/
git commit -m "Stage 2: V7.3 backbone ported with dict output for HLL energy"
```

---

## STAGE 3: V8.0 Losses + HLL Normalization

**Goal:** Implement the contrastive HLL loss and HLL energy normalization. These are the most impactful fixes — they address Failure Mode 1 (absolute HLL loss encoding codec signatures).

### Step 3.1: Create losses.py

```python
"""STF-Mamba V8.0 Loss Functions.

The key insight: MSE(HLL_real, 0) learns "real videos have zero flicker"
which is WRONG — real videos have compression noise that looks like flicker.
Contrastive loss instead learns "fakes have MORE flicker than reals" which
is scale-invariant across codecs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveHLLLoss(nn.Module):
    """Margin-based contrastive loss on HLL energy.
    
    L_HLL = max(0, margin - (||HLL_fake||_2 - ||HLL_real||_2))
    
    Only cares about RELATIVE difference, not absolute values.
    A noisy DFDC video has high HLL for both real and fake,
    but fake still has relatively more → loss still works.
    """
    def __init__(self, margin=1.0, normalize=True):
        super().__init__()
        self.margin = margin
        self.normalize = normalize
    
    def forward(self, hll_energy, labels, all_band_energy=None):
        """
        Args:
            hll_energy: (B,) HLL sub-band energy per sample
            labels: (B,) 0=real, 1=fake
            all_band_energy: (B,) total energy across all bands (for normalization)
        Returns:
            scalar loss
        """
        if self.normalize and all_band_energy is not None:
            # Normalize HLL relative to total wavelet energy
            hll_energy = hll_energy / (all_band_energy.sqrt() + 1e-8)
        
        real_mask = (labels == 0)
        fake_mask = (labels == 1)
        
        if real_mask.sum() == 0 or fake_mask.sum() == 0:
            return torch.tensor(0.0, device=hll_energy.device)
        
        real_energy = hll_energy[real_mask].mean()
        fake_energy = hll_energy[fake_mask].mean()
        
        loss = F.relu(self.margin - (fake_energy - real_energy))
        return loss


class STFV8Loss(nn.Module):
    """Combined loss for STF-Mamba V8.0.
    
    L_total = L_CE + lambda * L_HLL
    
    Where L_HLL is either:
    - MSE(HLL_real, 0) if use_contrastive_hll=False (V7.3 baseline)
    - Contrastive margin loss if use_contrastive_hll=True (V8.0 fix)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ce = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        
        if config.use_contrastive_hll:
            self.hll_loss = ContrastiveHLLLoss(
                margin=config.hll_loss_margin,
                normalize=config.use_hll_normalization
            )
        else:
            self.hll_loss = None  # Fall back to V7.3 MSE loss
        
        self.hll_weight = config.hll_loss_weight
    
    def forward(self, model_output, labels):
        """
        Args:
            model_output: dict with 'logits', 'hll_energy', optionally 'all_band_energy'
            labels: (B,) integer class labels
        Returns:
            dict with 'total', 'ce', 'hll' losses
        """
        logits = model_output['logits']
        ce_loss = self.ce(logits, labels)
        
        hll_energy = model_output.get('hll_energy', None)
        
        if hll_energy is not None and self.hll_weight > 0:
            if self.hll_loss is not None:
                # V8.0: Contrastive HLL loss
                all_band = model_output.get('all_band_energy', None)
                hll_loss = self.hll_loss(hll_energy, labels, all_band)
            else:
                # V7.3 fallback: MSE(HLL_real, 0)
                real_mask = (labels == 0)
                if real_mask.sum() > 0:
                    hll_loss = F.mse_loss(hll_energy[real_mask],
                                          torch.zeros_like(hll_energy[real_mask]))
                else:
                    hll_loss = torch.tensor(0.0, device=logits.device)
        else:
            hll_loss = torch.tensor(0.0, device=logits.device)
        
        total = ce_loss + self.hll_weight * hll_loss
        
        return {
            'total': total,
            'ce': ce_loss.detach(),
            'hll': hll_loss.detach(),
        }
```

### Step 3.2: Modify backbone_v73.py to output all_band_energy

In the DWT module output, add computation of total band energy (sum of L2 norms across all 8 sub-bands). This is needed by the contrastive loss when `use_hll_normalization=True`.

### Step 3.3: Validation Check

```python
import torch
from src.models.config import STFV8Config
from src.models.losses import STFV8Loss

cfg = STFV8Config.phase_a()
loss_fn = STFV8Loss(cfg)

# Simulate model output
fake_output = {
    'logits': torch.randn(8, 2),
    'hll_energy': torch.tensor([0.1, 0.2, 0.8, 0.9, 0.15, 0.3, 0.7, 0.85]),
    'all_band_energy': torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
}
labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])  # 4 real, 4 fake

losses = loss_fn(fake_output, labels)
print(f"Total: {losses['total']:.4f}, CE: {losses['ce']:.4f}, HLL: {losses['hll']:.4f}")
losses['total'].backward()  # Must not error
print("Stage 3 PASS: loss backward works")
```

### Step 3.4: Git commit

```bash
git add src/models/losses.py
git commit -m "Stage 3: Contrastive HLL loss + energy normalization"
```

---

## STAGE 4: Hydra Mixer + Mamer Blocks

**Goal:** Implement the Quasiseparable Hydra Mixer and Mamer (Mamba-Transformer) hybrid blocks. These address Failure Mode 2 (heuristic bidirectionality).

### Step 4.1: Create hydra_mixer.py

Core components:
1. `Conv1DSequenceMixer` — fallback when mamba_ssm not installed
2. `HydraQuasiseparableMixer` — single-pass bidirectional SSM
3. `MamerBlock` — SSM → Attention (replaces SSM → FFN)
4. `PNMamerStage` — 3 Hydra blocks + 1 Mamer block per stage

**Refer to the V8 modules already designed in previous conversation.**

Key implementation detail — the Hydra mixer formula:
```
M_QS(X) = shift(SS_fwd(X)) + flip(shift(SS_bwd(flip(X)))) + D·X
```

The `shift` operation is a 1-position causal shift that enables local context.

### Step 4.2: Validation Check

```python
import torch
from src.models.hydra_mixer import HydraQuasiseparableMixer, MamerBlock, PNMamerStage

# Test Hydra
mixer = HydraQuasiseparableMixer(d_model=128, d_state=16)
x = torch.randn(2, 32, 128)  # (B, T, D)
y = mixer(x)
assert y.shape == x.shape, f"Shape mismatch: {y.shape}"

# Test Mamer
mamer = MamerBlock(d_model=128, d_state=16, n_heads=4)
y = mamer(x)
assert y.shape == x.shape

# Test PNMamer Stage
stage = PNMamerStage(d_model=128, d_state=16, n_hydra=3, n_heads=4)
y = stage(x)
assert y.shape == x.shape

print(f"Hydra params: {sum(p.numel() for p in mixer.parameters())/1e3:.1f}K")
print("Stage 4 PASS")
```

### Step 4.3: Git commit

```bash
git add src/models/hydra_mixer.py
git commit -m "Stage 4: Quasiseparable Hydra + Mamer hybrid blocks"
```

---

## STAGE 5: W-SS Module + Full V8.0 Backbone

**Goal:** Implement Wavelet-Selective Scan and assemble the complete V8.0 backbone that switches between V7.3 and V8.0 based on config flags.

### Step 5.1: Create wss_module.py

Two streams:
- **Stream A (TemporalFlickerStream):** HLL → spatial pool → 1D temporal scan
- **Stream B (SpatialTextureStream):** HHH + HLH → channel-selective scan
- **Gated Cross-Merge:** F_fused = F_spatial ⊙ σ(F_temporal)

### Step 5.2: Create backbone_v8.py

The unified backbone that uses config flags:

```python
class STFMambaV8(nn.Module):
    def __init__(self, config: STFV8Config):
        super().__init__()
        self.config = config
        
        # Phase 1: Spatial backbone (always ConvNeXt V2 or V7.3 stem)
        # For now: V7.3 stem + stages (later: add ConvNeXt V2 option)
        self.stem = ...  # from V7.3
        
        # Phase 2: Frequency
        self.dwt = STF3DDWTModule(config)
        
        if config.use_wss:
            self.freq_processor = WaveletSelectiveScan(config)
        else:
            self.freq_processor = None  # Use V7.3 HLL gating
        
        # Phase 3: Temporal
        if config.use_hydra:
            # V8.0: PNMamer stages
            self.temporal_stages = nn.ModuleList([
                PNMamerStage(d_model=dim, d_state=config.d_state,
                            n_hydra=3 if config.use_mamer else config.stage_blocks[i],
                            n_heads=config.num_heads if config.use_mamer else 0)
                for i, dim in enumerate(stage_dims)
            ])
        else:
            # V7.3: Original DCConv + Mamba + Transformer stages
            self.temporal_stages = ...  # from V7.3
        
        # Phase 4: Classification
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(config.stage4_dim, config.num_classes)
        )
    
    def forward(self, x):
        # Returns dict with logits + hll_energy + all_band_energy
        ...
```

### Step 5.3: Validation Check

```python
import torch
from src.models.config import STFV8Config
from src.models.backbone_v8 import STFMambaV8

# Test V7.3 mode (no V8 flags)
cfg = STFV8Config.debug()
model = STFMambaV8(cfg)
x = torch.randn(2, 3, 4, 64, 64)
out = model(x)
print(f"V7.3 mode — logits: {out['logits'].shape}")

# Test Phase A mode
cfg = STFV8Config.phase_a()
cfg.num_frames = 4; cfg.img_size = 64  # Override for CPU test
cfg.stem_dim = 16; cfg.stage1_dim = 32; cfg.stage2_dim = 64
cfg.stage3_dim = 128; cfg.stage4_dim = 128
model = STFMambaV8(cfg)
out = model(x)
print(f"Phase A mode — logits: {out['logits'].shape}")

# Test Phase B mode
cfg = STFV8Config.phase_b()
# ... same overrides for CPU
model = STFMambaV8(cfg)
out = model(x)
print(f"Phase B mode — logits: {out['logits'].shape}")
n = sum(p.numel() for p in model.parameters())
print(f"Phase B params: {n/1e6:.2f}M")
print("Stage 5 PASS")
```

### Step 5.4: Git commit

```bash
git add src/models/wss_module.py src/models/backbone_v8.py
git commit -m "Stage 5: W-SS + unified V8.0 backbone with feature flags"
```

---

## STAGE 6: Training Script

**Goal:** Build the complete training script with budget tracking, checkpoint management, and YAML config support.

### Step 6.1: Create configs/v8_a100.yaml and v8_4090.yaml

```yaml
# v8_a100.yaml
data:
  batch_size: 16
  num_workers: 8
  image_size: 224
  num_frames: 32
  num_segments: 4
  frames_per_segment: 8
  data_root: /workspace/data  # or /runpod-volume/data

model:
  # V7.3 base dims
  stem_dim: 48
  stage1_dim: 96
  stage2_dim: 192
  stage3_dim: 384
  stage4_dim: 384
  dwt_channels: 64
  dwt_hidden_dim: 64
  d_state: 16
  d_conv: 4
  ssm_expand: 2
  num_heads: 8
  dropout: 0.3
  drop_path: 0.1

training:
  num_epochs: 70          # Phase A
  lr: 5.0e-4
  weight_decay: 0.05
  warmup_epochs: 5
  min_lr: 1.0e-6
  label_smoothing: 0.1
  max_grad_norm: 1.0
  gradient_accumulation_steps: 2
  amp_dtype: bfloat16
  cost_per_hour: 1.39
  max_budget: 5.0

# V8.0 feature flags
features:
  use_contrastive_hll: true
  use_compression_aug: true
  use_multiscale_drift: true
  use_hll_normalization: true
  use_hydra: false          # Enable in Phase B
  use_mamer: false
  use_wss: false
  use_adaptive_sdim: false
```

### Step 6.2: Create src/training/train.py

Port from V7.3 `train_final.py` with these changes:
1. Use `STFV8Config` instead of `Config`
2. Use `STFV8Loss` instead of `nn.CrossEntropyLoss`
3. Use `STFMambaV8` instead of `STFMambaV73`
4. Add budget tracker that stops training before exceeding limit
5. Flexible checkpoint loading (Phase A → Phase B compatible)
6. Load feature flags from YAML config

The training loop structure stays nearly identical — it works well.

### Step 6.3: Create src/training/budget_tracker.py

```python
class BudgetTracker:
    """Monitors GPU cost and stops before exceeding budget."""
    def __init__(self, cost_per_hour, max_budget, safety_margin=0.10):
        self.cost_per_hour = cost_per_hour
        self.max_budget = max_budget
        self.safety = safety_margin
        self.start_time = time.time()
    
    def check(self):
        elapsed_hrs = (time.time() - self.start_time) / 3600
        spent = elapsed_hrs * self.cost_per_hour
        remaining = self.max_budget - spent
        safe_hours = remaining / self.cost_per_hour * (1 - self.safety)
        return {
            'spent': spent,
            'remaining': remaining,
            'safe_hours': safe_hours,
            'should_stop': remaining < (self.cost_per_hour * 0.1),  # < 6 min left
        }
```

### Step 6.4: Create scripts/verify_data.py

Quick script that checks your preprocessed data is accessible:

```python
"""Verify preprocessed data exists and is readable."""
# Checks: crops/*.npz exist, landmarks/*.npz exist, split files exist
# Reports: total samples, missing files, sample shapes
```

### Step 6.5: Create scripts/setup_pod.sh

```bash
#!/bin/bash
# STF-Mamba V8.0 — Pod Setup (assumes preprocessed data available)
# 
# Two modes:
#   1. Network volume: data at /runpod-volume/data (fast, recommended)
#   2. Ephemeral: data at /workspace/data (from previous run)
#
# USAGE:
#   cd /workspace && git clone https://github.com/AbdelRahman-Madboly/STF-Mamba_V8.0.git
#   cd STF-Mamba_V8.0 && bash scripts/setup_pod.sh
```

Contents:
1. Detect GPU → auto-configure batch size / accumulation steps
2. Install Python deps (torch, mamba_ssm with fallback, albumentations, dlib)
3. Verify data paths (check /runpod-volume/data first, then /workspace/data)
4. Create symlinks so code finds data regardless of location
5. Run verify_data.py
6. Print cost estimate for current GPU

### Step 6.6: Validation Check (dry run)

```bash
# On your local machine (CPU, dummy data):
python -m src.training.train --config configs/v8_debug.yaml --dry-run --epochs 2
# Should complete 2 epochs on random data, log to console, save dummy checkpoint
```

### Step 6.7: Git commit

```bash
git add configs/ scripts/ src/training/ src/utils/
git commit -m "Stage 6: training script with budget tracker + pod setup"
git push
```

---

## STAGE 7: RunPod Setup + Data Mount

**Goal:** Get your RunPod pod ready with code and data.

### Step 7.1: Start RunPod pod

Choose GPU:
- **A100 80GB ($1.39/hr):** Faster per-epoch, but tight budget
- **RTX 4090 24GB ($0.39/hr):** Slower per-epoch, but 3× more total epochs

Use PyTorch 2.1+ template. Enable network volume if you have one.

### Step 7.2: Clone and setup

```bash
cd /workspace
git clone https://github.com/AbdelRahman-Madboly/STF-Mamba_V8.0.git
cd STF-Mamba_V8.0
bash scripts/setup_pod.sh
```

### Step 7.3: Verify data

```bash
python scripts/verify_data.py
```

Expected output:
```
✓ Crops directory: /workspace/data/preprocessed/crops/ (N files)
✓ Landmarks directory: /workspace/data/preprocessed/landmarks/ (N files)
✓ Split files found: train.json, val.json, test.json
✓ Sample check: crops shape OK, landmarks shape OK
✓ Ready to train!
```

### Step 7.4: CRITICAL — Save data to network volume

If your data is on ephemeral storage and you have a network volume:

```bash
# This takes ~10-20 minutes but saves $2+ on every future run
cp -r /workspace/data/preprocessed /runpod-volume/data/
cp -r /workspace/STF-Mamba_V8.0/splits /runpod-volume/data/
echo "Data saved to network volume!"
```

---

## STAGE 8: Phase A Training

**Goal:** Train V7.3 architecture + all three critical fixes. Target: break 51% → 70-85% AUC.

### Step 8.1: Start Phase A

```bash
cd /workspace/STF-Mamba_V8.0

# For A100:
python -m src.training.train --config configs/v8_a100.yaml --phase a

# For RTX 4090:
python -m src.training.train --config configs/v8_4090.yaml --phase a
```

### Step 8.2: Monitor training

Watch for:
- **Epoch 1-5 (warmup):** Loss should decrease from ~0.7 to ~0.4
- **Epoch 5-20:** Val AUC should climb past 90% on SBI validation
- **Epoch 20-70:** Watch for convergence, AUC should plateau ~98-99%

The budget tracker will warn you when approaching the limit.

### Step 8.3: Quick evaluation

```bash
python -m src.training.evaluate \
    --checkpoint outputs/checkpoints/best.pth \
    --config configs/v8_a100.yaml
```

**Decision point after Phase A:**
- AUC > 75% on real deepfakes → **Phase B will help, proceed**
- AUC 60-75% → **Phase A fixes working, Phase B should push to 85%+**
- AUC < 60% → **Debug augmentation pipeline first** (return to Opus for diagnosis)
- AUC > 85% → **Phase A alone might be sufficient, evaluate further before Phase B**

---

## STAGE 9: Evaluation + Phase B

### Step 9.1: Cross-dataset evaluation

```bash
# Evaluate on all FF++ manipulation types
python -m src.training.evaluate \
    --checkpoint outputs/checkpoints/best.pth \
    --methods Deepfakes Face2Face FaceSwap NeuralTextures \
    --config configs/v8_a100.yaml
```

### Step 9.2: Phase B (if budget remains)

Edit the YAML config to enable V8.0 flags:

```yaml
features:
  use_hydra: true
  use_mamer: true
  use_wss: true
  use_adaptive_sdim: true
```

Then:
```bash
python -m src.training.train \
    --config configs/v8_a100.yaml \
    --phase b \
    --resume outputs/checkpoints/best.pth
```

Phase B loads the Phase A checkpoint with flexible key matching — it skips incompatible keys from the architecture change and initializes new modules from scratch.

### Step 9.3: Final evaluation

```bash
python -m src.training.evaluate \
    --checkpoint outputs/checkpoints/phase_b_best.pth \
    --full-report
```

---

## Budget Summary

### A100 80GB ($1.39/hr)

| Activity | Time | Cost |
|----------|------|------|
| Setup + verify | 10 min | $0.23 |
| Phase A (50 epochs) | 2.5 hrs | $3.48 |
| Evaluation | 15 min | $0.35 |
| **Total** | **~3 hrs** | **$4.06** |
| Remaining for Phase B | ~40 min | $0.94 |

### RTX 4090 ($0.39/hr)

| Activity | Time | Cost |
|----------|------|------|
| Setup + verify | 10 min | $0.07 |
| Phase A (70 epochs) | 5.4 hrs | $2.11 |
| Evaluation | 15 min | $0.10 |
| Phase B (50 epochs) | 5.3 hrs | $2.07 |
| **Total** | **~11 hrs** | **$4.35** |

---

## File Creation Order for Cursor

When working in Cursor, create files in this exact order:

```
1. .gitignore
2. requirements.txt
3. src/__init__.py (empty)
4. src/models/__init__.py (empty)
5. src/models/config.py              ← STAGE 0
6. src/data/__init__.py (empty)
7. src/data/blend.py                 ← STAGE 1 (copy from SBI reference)
8. src/data/funcs.py                 ← STAGE 1 (copy from SBI reference)
9. src/data/augmentations.py         ← STAGE 1
10. src/data/video_sbi_dataset.py    ← STAGE 1
11. src/models/wavelet_constants.py  ← STAGE 2 (copy from V7.3)
12. src/models/drop_path.py          ← STAGE 2 (copy from V7.3)
13. src/models/dwt_3d.py             ← STAGE 2 (copy from V7.3)
14. src/models/dcconv_3d.py          ← STAGE 2 (copy from V7.3)
15. src/models/mamba_blocks.py       ← STAGE 2 (copy from V7.3)
16. src/models/transformer_blocks.py ← STAGE 2 (copy from V7.3)
17. src/models/backbone_v73.py       ← STAGE 2 (port from V7.3)
18. src/models/losses.py             ← STAGE 3
19. src/models/hydra_mixer.py        ← STAGE 4
20. src/models/wss_module.py         ← STAGE 5
21. src/models/backbone_v8.py        ← STAGE 5
22. src/utils/__init__.py (empty)
23. src/utils/logging.py             ← STAGE 6
24. src/utils/checkpoint.py          ← STAGE 6
25. src/training/__init__.py (empty)
26. src/training/budget_tracker.py   ← STAGE 6
27. src/training/train.py            ← STAGE 6
28. src/training/evaluate.py         ← STAGE 6
29. configs/v8_a100.yaml             ← STAGE 6
30. configs/v8_4090.yaml             ← STAGE 6
31. scripts/setup_pod.sh             ← STAGE 6
32. scripts/verify_data.py           ← STAGE 6
33. scripts/preprocess.py            ← STAGE 6 (optional, if need to re-preprocess)
34. README.md                        ← STAGE 6
```

---

## What to Tell Cursor at Each Stage

### Stage 0 prompt:
> "Create the project scaffold for STF-Mamba V8.0. Start with .gitignore, requirements.txt, and src/models/config.py with the STFV8Config dataclass that has feature flags for Phase A (loss+augmentation fixes) and Phase B (Hydra+Mamer+WSS architecture). Include class methods phase_a(), phase_b(), debug(), and describe()."

### Stage 1 prompt:
> "Create the Video-SBI data pipeline. Port blend.py and funcs.py from the SBI reference code. Then create augmentations.py with CompressionAugmentation (random H.264 QP 23-40), MultiScaleBrownianDrift (0.5-5px log-normal), and TPSMaskDeformation. Finally create video_sbi_dataset.py that adapts V7.3's CachedSBIDataset to use these V8 augmentations controlled by STFV8Config flags."

### Stage 2 prompt:
> "Port the V7.3 backbone modules (dwt_3d.py, dcconv_3d.py, mamba_blocks.py, transformer_blocks.py, drop_path.py, wavelet_constants.py) into src/models/. Fix imports to use STFV8Config. Create backbone_v73.py that returns a dict {'logits': ..., 'hll_energy': ..., 'all_band_energy': ...} instead of just logits."

### Stage 3 prompt:
> "Create src/models/losses.py with ContrastiveHLLLoss (margin-based, with optional HLL normalization by total band energy) and STFV8Loss (combines CE + contrastive HLL, controlled by config flags, falls back to V7.3 MSE loss when flag is off)."

### Stage 4 prompt:
> "Create src/models/hydra_mixer.py with: (1) Conv1DSequenceMixer fallback, (2) HydraQuasiseparableMixer implementing M_QS(X) = shift(SS_fwd(X)) + flip(shift(SS_bwd(flip(X)))) + D·X, (3) MamerBlock (SSM→Attention replacing SSM→FFN), (4) PNMamerStage (3 Hydra + 1 Mamer per stage)."

### Stage 5 prompt:
> "Create src/models/wss_module.py with dual-stream WaveletSelectiveScan (Stream A: HLL temporal flicker scan, Stream B: HHH+HLH spatial texture scan, Gated Cross-Merge). Then create backbone_v8.py — unified model that uses V7.3 stages when Hydra/Mamer/WSS flags are off, and V8.0 stages when on."

### Stage 6 prompt:
> "Create the training infrastructure: (1) src/training/train.py adapted from V7.3 train_final.py using STFV8Loss and STFMambaV8, with Phase A/B support and flexible checkpoint loading; (2) budget_tracker.py; (3) YAML configs for A100 and 4090; (4) scripts/setup_pod.sh for RunPod environment setup; (5) scripts/verify_data.py for data integrity check."

---

## Recovery Procedures

### If training diverges (loss goes to NaN):
1. Reduce learning rate by 5× in YAML
2. Increase warmup_epochs to 10
3. Check if compression augmentation is too aggressive (raise qp_min to 28)

### If GPU OOM:
1. Reduce batch_size by half, double grad_accum
2. Enable gradient checkpointing in config
3. Reduce num_frames from 32 to 16

### If Phase A AUC < 55% (no improvement):
1. Disable all V8 augmentations, train baseline V7.3 first
2. Verify data pipeline: save sample batches, visually inspect SBI quality
3. Check that randaffine is applied (this was the V7.3 critical bug)

### If Phase B crashes on checkpoint load:
1. Check error message — which key is incompatible
2. Use `--no-resume` flag to start Phase B from scratch (less ideal)
3. Or manually filter checkpoint keys in checkpoint.py

---

## Success Criteria

| Metric | V7.3 Baseline | Phase A Target | Phase B Target |
|--------|---------------|----------------|----------------|
| SBI Val AUC | 98.57% | ≥98% | ≥98% |
| Real FF++ AUC | 51.55% | ≥70% | ≥85% |
| Deepfakes AUC | 59.5% | ≥80% | ≥90% |
| Face2Face AUC | 49.6% | ≥65% | ≥80% |
| FaceSwap AUC | 51.8% | ≥70% | ≥85% |
| NeuralTextures AUC | 45.3% | ≥60% | ≥75% |
