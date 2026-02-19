# STF-Mamba V8.0 — Cursor Prompts (Copy-Paste Ready)

> **How to use:** Open Cursor, paste each prompt into chat in order. Wait for Cursor to finish before pasting the next one. Each prompt gives Cursor full context + specific task.

---

## PROMPT 0: Project Context + Scaffold

```
I'm building STF-Mamba V8.0, a video deepfake detector. Here's the full context:

## PROBLEM
My V7.3 model gets 98.57% AUC on SBI (self-blended image) validation but only 51.55% AUC on real deepfakes — it memorized training-specific artifacts instead of learning universal forgery traces.

## ROOT CAUSES (diagnosed)
1. HLL Consistency Loss `MSE(HLL_real, 0)` encodes codec-specific compression signatures rather than forgery traces
2. Heuristic Bi-Mamba (two independent SSMs concatenated) allows each branch to learn dataset shortcuts independently
3. Fixed ±2px Brownian drift in Video-SBI is too narrow — Mamba state memorizes this exact pattern

## SOLUTION: V8.0 Architecture
Phase A (same V7.3 architecture + 3 fixes):
- Contrastive HLL Loss: `max(0, margin - (||HLL_fake|| - ||HLL_real||))` — relative, not absolute
- Compression Augmentation: random H.264 re-compression QP 23-40 during training
- Multi-Scale Brownian Drift: step sizes 0.5-5.0px (log-normal) with varied velocity profiles

Phase B (architecture upgrades):
- Quasiseparable Hydra Mixer (native bidirectional SSM, single-pass)
- Mamer Blocks (SSM → Attention replacing SSM → FFN)
- Wavelet-Selective Scan (frequency-native dual-stream processing)

## DATA FORMAT (already preprocessed on RunPod network volume)
```
/runpod-volume/data/FaceForensics++/
  original_sequences/youtube/raw/
    frames/{video_id}/*.png          # Extracted face crops
    landmarks/{video_id}/*.npy       # 81-point dlib landmarks
    retina/{video_id}/*.npy          # RetinaFace bounding boxes
  train.json, val.json, test.json    # FF++ official splits
```

SBI fakes are generated ON-THE-FLY during training (not pre-cached). For V8.0, we extend single-frame SBI to VIDEO-SBI: load 32 consecutive frames from a real video, generate a self-blended mask, apply Brownian drift across frames to create "swimming" artifacts.

## TRAINING
- GPU: A100 80GB ($1.39/hr)
- Budget: ~$5 total
- Phase A: 70 epochs with V7.3 architecture + fixes
- Phase B: 50 epochs enabling Hydra + Mamer + W-SS

## REPO
https://github.com/AbdelRahman-Madboly/STF-Mamba_V8.0.git
Local: C:\Dan_WS\STF-Mamba_V8.0

---

Now create the project scaffold. Create these files:

1. `.gitignore` — exclude *.pth, *.pt, *.tar, data/, __pycache__/, .vscode/, *.mp4, *.npz, *.npy, wandb/

2. `requirements.txt`:
```
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
imutils>=0.5.4
```

3. Empty `__init__.py` files in: `src/`, `src/models/`, `src/data/`, `src/training/`, `src/utils/`

4. `src/models/config.py` — STFV8Config dataclass with:
   - Input params: in_channels=3, num_classes=2, num_frames=32, img_size=224
   - Backbone dims: stem_dim=48, stage1_dim=96, stage2_dim=192, stage3_dim=384, stage4_dim=384
   - Block counts: stage1_blocks=2, stage2_blocks=2, stage3_blocks=4, stage4_blocks=2
   - DWT: dwt_channels=64, dwt_hidden_dim=64
   - SSM: d_state=16, d_conv=4, ssm_expand=2
   - Attention: num_heads=8, mlp_ratio=2.0
   - Regularization: dropout=0.3, drop_path=0.1
   - DCConv: dcconv_kernel_length=9
   - Feature flags (all bool, default False): use_contrastive_hll, use_compression_aug, use_multiscale_drift, use_hll_normalization, use_hydra, use_mamer, use_wss, use_adaptive_sdim
   - Training: lr=5e-4, weight_decay=0.05, warmup_epochs=5, min_lr=1e-6, label_smoothing=0.1, hll_loss_weight=0.1, hll_loss_margin=1.0, max_grad_norm=1.0, grad_accum=2, amp_dtype="bfloat16"
   - Budget: cost_per_hour=1.39, max_budget=5.0
   - Data: data_root="/runpod-volume/data", num_workers=8, batch_size=16
   - Class methods: phase_a() enables contrastive_hll + compression_aug + multiscale_drift + hll_normalization; phase_b() enables all phase_a + hydra + mamer + wss + adaptive_sdim with lr=1e-4; debug() uses tiny dims for CPU testing
   - describe() method that prints active features

5. Directory structure should be:
```
STF-Mamba_V8.0/
├── configs/
│   └── v8_a100.yaml
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── config.py
│   ├── data/
│   │   └── __init__.py
│   ├── training/
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── scripts/
├── splits/
├── .gitignore
└── requirements.txt
```

Don't create the YAML config yet — we'll do that in Stage 6.
```

---

## PROMPT 1A: SBI Reference Utilities (blend + funcs)

```
Now create the SBI data utilities. These are adapted from the official SBI paper code (Shiohara & Yamasaki, CVPR 2022).

Create `src/data/blend.py`:
```python
"""SBI blending functions — adapted from official SBI implementation."""
import cv2
import numpy as np
import random

def alpha_blend(source, target, mask):
    """Standard alpha blending."""
    mask_blured = get_blend_mask(mask)
    img_blended = (mask_blured * source + (1 - mask_blured) * target)
    return img_blended, mask_blured

def dynamic_blend(source, target, mask):
    """Dynamic blend with random blend ratio (favors full blend)."""
    mask_blured = get_blend_mask(mask)
    blend_list = [0.25, 0.5, 0.75, 1, 1, 1]
    blend_ratio = blend_list[np.random.randint(len(blend_list))]
    mask_blured *= blend_ratio
    img_blended = (mask_blured * source + (1 - mask_blured) * target)
    return img_blended, mask_blured

def get_blend_mask(mask):
    """Create soft blending mask with random blur kernels."""
    H, W = mask.shape
    size_h = np.random.randint(192, 257)
    size_w = np.random.randint(192, 257)
    mask = cv2.resize(mask, (size_w, size_h))
    kernel_1 = random.randrange(5, 26, 2)
    kernel_1 = (kernel_1, kernel_1)
    kernel_2 = random.randrange(5, 26, 2)
    kernel_2 = (kernel_2, kernel_2)
    mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
    mask_blured = mask_blured / (mask_blured.max() + 1e-8)
    mask_blured[mask_blured < 1] = 0
    mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, np.random.randint(5, 46))
    mask_blured = mask_blured / (mask_blured.max() + 1e-8)
    mask_blured = cv2.resize(mask_blured, (W, H))
    return mask_blured.reshape((mask_blured.shape + (1,)))
```

Create `src/data/funcs.py`:
```python
"""Utility functions — adapted from official SBI implementation."""
import numpy as np
import cv2
import albumentations as alb
import json
from glob import glob
import os

def load_json(path):
    with open(path, mode="r") as f:
        return json.load(f)

def IoUfrom2bboxes(boxA, boxB):
    """Compute IoU between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def crop_face(img, landmark=None, bbox=None, margin=False, crop_by_bbox=True, abs_coord=False, only_img=False, phase='train'):
    """Crop face region from image using landmarks or bounding box.
    
    This is the EXACT logic from the SBI paper — do not modify.
    """
    assert phase in ['train', 'val', 'test']
    H, W = len(img), len(img[0])
    assert landmark is not None or bbox is not None
    
    if crop_by_bbox:
        x0, y0 = bbox[0]
        x1, y1 = bbox[1]
        w = x1 - x0
        h = y1 - y0
        w0_margin = w / 4
        w1_margin = w / 4
        h0_margin = h / 4
        h1_margin = h / 4
    else:
        x0, y0 = landmark[:68, 0].min(), landmark[:68, 1].min()
        x1, y1 = landmark[:68, 0].max(), landmark[:68, 1].max()
        w = x1 - x0
        h = y1 - y0
        w0_margin = w / 8
        w1_margin = w / 8
        h0_margin = h / 2
        h1_margin = h / 5
    
    if margin:
        w0_margin *= 4
        w1_margin *= 4
        h0_margin *= 2
        h1_margin *= 2
    elif phase == 'train':
        w0_margin *= (np.random.rand() * 0.6 + 0.2)
        w1_margin *= (np.random.rand() * 0.6 + 0.2)
        h0_margin *= (np.random.rand() * 0.6 + 0.2)
        h1_margin *= (np.random.rand() * 0.6 + 0.2)
    else:
        w0_margin *= 0.5
        w1_margin *= 0.5
        h0_margin *= 0.5
        h1_margin *= 0.5
    
    y0_new = max(0, int(y0 - h0_margin))
    y1_new = min(H, int(y1 + h1_margin) + 1)
    x0_new = max(0, int(x0 - w0_margin))
    x1_new = min(W, int(x1 + w1_margin) + 1)
    
    img_cropped = img[y0_new:y1_new, x0_new:x1_new]
    if landmark is not None:
        landmark_cropped = np.zeros_like(landmark)
        for i, (p, q) in enumerate(landmark):
            landmark_cropped[i] = [p - x0_new, q - y0_new]
    else:
        landmark_cropped = None
    if bbox is not None:
        bbox_cropped = np.zeros_like(bbox)
        for i, (p, q) in enumerate(bbox):
            bbox_cropped[i] = [p - x0_new, q - y0_new]
    else:
        bbox_cropped = None
    
    if only_img:
        return img_cropped
    if abs_coord:
        return img_cropped, landmark_cropped, bbox_cropped, (y0 - y0_new, x0 - x0_new, y1_new - y1, x1_new - x1), y0_new, y1_new, x0_new, x1_new
    else:
        return img_cropped, landmark_cropped, bbox_cropped, (y0 - y0_new, x0 - x0_new, y1_new - y1, x1_new - x1)


class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
    """Random downscale augmentation from SBI paper."""
    def apply(self, img, **params):
        return self.randomdownscale(img)
    
    def randomdownscale(self, img):
        keep_input_shape = True
        H, W, C = img.shape
        ratio_list = [2, 4]
        r = ratio_list[np.random.randint(len(ratio_list))]
        img_ds = cv2.resize(img, (int(W / r), int(H / r)), interpolation=cv2.INTER_NEAREST)
        if keep_input_shape:
            img_ds = cv2.resize(img_ds, (W, H), interpolation=cv2.INTER_LINEAR)
        return img_ds


def init_ff(data_root, phase, level='frame', n_frames=8):
    """Initialize FaceForensics++ file lists.
    
    Args:
        data_root: root path containing FaceForensics++/
        phase: 'train', 'val', or 'test'
        level: 'frame' or 'video'
        n_frames: max frames per video
    Returns:
        image_list, label_list
    """
    dataset_path = os.path.join(data_root, 'FaceForensics++/original_sequences/youtube/raw/frames/')
    image_list = []
    label_list = []
    
    folder_list = sorted(glob(dataset_path + '*'))
    filelist = []
    split_path = os.path.join(data_root, f'FaceForensics++/{phase}.json')
    list_dict = json.load(open(split_path, 'r'))
    for i in list_dict:
        filelist += i
    folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]
    
    if level == 'video':
        label_list = [0] * len(folder_list)
        return folder_list, label_list
    
    for i in range(len(folder_list)):
        images_temp = sorted(glob(folder_list[i] + '/*.png'))
        if n_frames < len(images_temp):
            images_temp = [images_temp[round(j)] for j in np.linspace(0, len(images_temp) - 1, n_frames)]
        image_list += images_temp
        label_list += [0] * len(images_temp)
    
    return image_list, label_list
```

IMPORTANT: The `init_ff` function now accepts `data_root` as a parameter instead of hardcoding paths. This lets us point to `/runpod-volume/data` or `/workspace/data` depending on the pod setup.
```

---

## PROMPT 1B: V8.0 Augmentations

```
Now create the V8.0 augmentation upgrades. These address the root cause of V7.3's overfitting.

Create `src/data/augmentations.py` with three classes:

1. `CompressionAugmentation` — Random H.264 re-compression
   - Purpose: Force model to learn HLL patterns that survive varying compression levels
   - Implementation: Given a list of frames (uint8 BGR), randomly re-compress with H.264 at QP sampled from [23, 40]
   - Use ffmpeg subprocess for real H.264 compression
   - Fallback: per-frame JPEG compression if ffmpeg fails
   - probability parameter controls how often compression is applied (default 0.5)

2. `MultiScaleBrownianDrift` — Replaces fixed ±2px drift
   - Purpose: Prevent Mamba from memorizing one specific drift pattern
   - Implementation:
     - Sample step_size from log-normal distribution between step_range=(0.5, 5.0)
     - Randomly choose velocity profile: 'smooth' (Gaussian steps), 'jittery' (uniform steps), 'accelerating' (increasing scale), 'oscillating' (sinusoidal + noise)
     - Apply cumulative drift via cv2.warpAffine translation per frame
     - Clamp cumulative magnitude to max_cumulative=8.0
   - Input: base_mask (H, W) float32 [0,1], n_frames int
   - Output: list of n_frames drifted masks

3. `TPSMaskDeformation` — Optional non-rigid deformation
   - Purpose: Create locally varying boundary motion (not just translation)
   - Implementation: Use cv2.createThinPlateSplineShapeTransformer with random control point displacements
   - probability=0.3 (only applied sometimes)

All three classes should work independently and be composable. The dataset class will call them based on config flags.
```

---

## PROMPT 1C: Video-SBI Dataset

```
Now create the main dataset class that extends SBI to video sequences with V8.0 augmentations.

Create `src/data/video_sbi_dataset.py`:

This class adapts the original SBI_Dataset (single frame) to produce VIDEO clips of 32 frames with temporally drifting masks. The V8.0 augmentations (compression, multi-scale drift) are controlled by the STFV8Config feature flags.

## Data format on disk
The preprocessed data on the RunPod volume is:
```
{data_root}/FaceForensics++/original_sequences/youtube/raw/
  frames/{video_id}/000.png, 001.png, ..., 031.png   (32 frames per video)
  landmarks/{video_id}/000.npy, 001.npy, ...          (81-point landmarks per frame)  
  retina/{video_id}/000.npy, 001.npy, ...             (RetinaFace bboxes per frame)
```
Split files: `{data_root}/FaceForensics++/train.json`, `val.json`, `test.json`
Each split file contains a list of lists of video IDs (e.g. [["000", "001"], ["002", "003"], ...])

## Class: VideoSBIDataset(Dataset)

### __init__(self, phase, config: STFV8Config):
- Load video folder list using init_ff(config.data_root, phase, level='video')
- Filter to only videos that have both landmarks/ and retina/ files
- Store config, phase, image_size, n_frames=config.num_frames
- Initialize transforms (same as SBI paper):
  - source_transforms: RGBShift, HueSaturationValue, RandomBrightnessContrast, RandomDownScale/Sharpen
  - train_transforms: RGBShift, HueSaturationValue, RandomBrightnessContrast, ImageCompression(40-100)
- Initialize randaffine transforms (CRITICAL — this is what makes SBI work):
  - Affine: translate_percent x=(-0.03,0.03), y=(-0.015,0.015), scale=(0.95, 1/0.95)
  - ElasticTransform: alpha=50, sigma=7
- Initialize V8 augmentations based on config flags:
  - if config.use_compression_aug: self.compression_aug = CompressionAugmentation(qp_range=(23, 40))
  - if config.use_multiscale_drift: self.drift = MultiScaleBrownianDrift(step_range=(0.5, 5.0))
  - else: self.drift = MultiScaleBrownianDrift(step_range=(2.0, 2.0))  # V7.3 behavior: fixed 2px

### __len__: return len(video_list) * 2 (each video produces both real and fake)

### __getitem__(self, idx):
- video_idx = idx // 2
- is_fake = (idx % 2 == 1)
- Load n_frames consecutive frames from the video folder
- For each frame: load the image (.png), landmark (.npy), retina bbox (.npy)
- Match bbox to landmark using IoU (same as SBI reference)
- Reorder landmarks (81-point to match SBI convention, the reorder_landmark method)
- If is_fake:
  - Generate mask from first frame landmarks using convexHull
  - Apply Brownian drift to get n_frames drifted masks
  - For each frame:
    - Apply source_transforms to source or target (50% chance each)
    - Apply randaffine (Affine + ElasticTransform) to source image AND mask — THIS IS CRITICAL
    - Apply dynamic_blend to create fake frame
  - If config.use_compression_aug: apply compression augmentation to all frames
- If is_real: just use the original frames
- Apply train_transforms (if training phase) to all frames consistently
- Crop face, resize to (img_size, img_size), stack to (3, n_frames, H, W) tensor
- Return dict: {'video': tensor, 'label': 0 or 1}

### collate_fn: standard batch collation

### Key implementation notes:
- Wrap __getitem__ in try/except with random retry (up to 10 attempts) — same pattern as SBI reference
- The randaffine MUST be applied to both the source image AND the mask together — this creates the visible blend boundary that the model needs to learn. Without it, fake ≈ real and the model can't learn.
- For video mode: apply crop_face parameters from FIRST frame to ALL frames (consistent crop across video)
- The hflip augmentation should be applied consistently to all frames (50% chance during training)

Import from: src.data.blend (dynamic_blend), src.data.funcs (crop_face, IoUfrom2bboxes, RandomDownScale, init_ff), src.data.augmentations (CompressionAugmentation, MultiScaleBrownianDrift)
```

---

## PROMPT 2: Port V7.3 Backbone

```
Now port the V7.3 backbone modules. I'll provide the existing code — adapt it to use STFV8Config and ensure the backbone outputs a dict with HLL energy.

The V7.3 architecture processes (B, 3, T, H, W) video through 4 phases:
Phase 1: 3D-DWT (Symlet-2) → 8 sub-bands → HLL attention gating
Phase 2: Stem + Stage 1-2 with 3D-DCConv blocks + HLL gating: F_out = DCConv(F) × (1 + σ(HLL_attention))
Phase 3: Stage 3 with Bi-directional Mamba (forward + backward SSM, concatenated + gated projection)
Phase 4: Stage 4 with Transformer self-attention for global consistency

Port these files from the existing stf_mamba_backbone/ directory into src/models/:

1. `src/models/wavelet_constants.py` — Symlet-2 filter coefficients (copy as-is)
2. `src/models/drop_path.py` — DropPath stochastic depth (copy as-is)
3. `src/models/dwt_3d.py` — STF3DDWTModule that decomposes (B, C, T, H, W) into 8 sub-bands and extracts HLL attention. MODIFICATION: also return total band energy (sum of L2 norms across all 8 sub-bands) — needed for HLL normalization in V8 contrastive loss.
4. `src/models/dcconv_3d.py` — DCConv3DBlock with learnable offsets (Δt, Δy, Δx) for tracking swim boundaries (copy as-is, fix imports to use STFV8Config)
5. `src/models/mamba_blocks.py` — MambaVisionMixer with forward + backward SSM (copy as-is, fix imports)
6. `src/models/transformer_blocks.py` — TransformerBlock for stage 4 self-attention (copy as-is, fix imports)
7. `src/models/backbone_v73.py` — Main STFMambaV73 model that assembles all phases. MODIFICATION: forward() returns dict:
   ```python
   return {
       'logits': logits,           # (B, 2) classification output
       'hll_energy': hll_energy,   # (B,) per-sample HLL sub-band energy
       'all_band_energy': all_band_energy,  # (B,) per-sample total wavelet energy
   }
   ```

For all files: change `from .config import STFMambaConfig` to `from .config import STFV8Config`

The model input is (B, 3, T, H, W) where T=num_frames, H=W=img_size.
The model output is the dict above.

If mamba_ssm package is not available, the mamba_blocks.py should have a fallback that uses Conv1d as a sequence mixer (causal convolution approximation). This allows CPU testing.

Test: After creating all files, this should work:
```python
import torch
from src.models.config import STFV8Config
from src.models.backbone_v73 import STFMambaV73
cfg = STFV8Config.debug()
model = STFMambaV73(cfg)
x = torch.randn(2, 3, cfg.num_frames, cfg.img_size, cfg.img_size)
out = model(x)
assert out['logits'].shape == (2, 2)
assert out['hll_energy'].shape == (2,)
```
```

---

## PROMPT 3: Contrastive HLL Loss

```
Create the V8.0 loss functions in `src/models/losses.py`.

## Background
V7.3 used MSE(HLL_real, 0) which teaches "real videos have zero temporal flicker" — this is WRONG because real videos have compression noise that looks like flicker. The model learned to detect FF++ compression signature instead of forgery.

## New losses:

### 1. ContrastiveHLLLoss
```python
L_HLL = max(0, margin - (mean(||HLL_fake||) - mean(||HLL_real||)))
```
- Only cares about RELATIVE difference between real and fake HLL energy
- Scale-invariant: a noisy DFDC video has high HLL for both, but fake still has relatively more
- margin is learnable, default 1.0
- If use_hll_normalization: normalize HLL by sqrt(sum of all band energies + eps) before computing loss
- If batch has no real or no fake samples, return 0.0

### 2. STFV8Loss (combined loss)
```python
L_total = L_CE + λ × L_HLL
```
Where:
- L_CE = CrossEntropyLoss with label_smoothing from config
- L_HLL = ContrastiveHLLLoss if config.use_contrastive_hll else MSE(HLL_real, 0) (V7.3 fallback)
- λ = config.hll_loss_weight (default 0.1)

Input: model_output dict (has 'logits', 'hll_energy', 'all_band_energy'), labels tensor
Output: dict with 'total', 'ce', 'hll' loss values (total has grad, ce and hll are detached for logging)

Test:
```python
import torch
from src.models.config import STFV8Config
from src.models.losses import STFV8Loss
cfg = STFV8Config.phase_a()
loss_fn = STFV8Loss(cfg)
out = {'logits': torch.randn(8, 2), 'hll_energy': torch.rand(8), 'all_band_energy': torch.ones(8)}
labels = torch.tensor([0,0,0,0,1,1,1,1])
losses = loss_fn(out, labels)
losses['total'].backward()  # must not error
```
```

---

## PROMPT 4: Hydra Mixer + Mamer Blocks

```
Create the V8.0 temporal modeling modules in `src/models/hydra_mixer.py`.

These replace V7.3's heuristic Bi-Mamba (two independent SSMs concatenated) with mathematically principled bidirectionality.

## Components to implement:

### 1. Conv1DSequenceMixer (fallback)
Simple causal Conv1d mixer for when mamba_ssm is not installed. Allows CPU testing.
- Input: (B, T, D) → Output: (B, T, D)
- Uses depth-wise Conv1d with SiLU activation

### 2. HydraQuasiseparableMixer
Native bidirectional SSM in a single pass:
```
M_QS(X) = shift(SS_fwd(X)) + flip(shift(SS_bwd(flip(X)))) + D·X
```
Where:
- SS_fwd: standard Mamba forward scan
- SS_bwd: standard Mamba backward scan (flip input, scan, flip output)
- shift: 1-position causal shift for local context
- D: diagonal matrix for instant (same-position) interactions
- Shared input projections for both directions (~50% fewer params than Bi-Mamba)

Implementation:
- Input projection: x → (z, x_proj) via single linear layer, split
- x_proj → Conv1d → SiLU → split into (x_fwd, x_bwd)
- Forward SSM on x_fwd, backward SSM on x_bwd (flip, scan, flip)
- Shift both by 1 position
- Combine: y = shift(y_fwd) + shift(y_bwd) + D * x_proj
- Output: y * SiLU(z) → output projection

Try to use mamba_ssm.modules.mamba_simple.Mamba for the SSM. If not available, use Conv1DSequenceMixer as fallback.

### 3. MamerBlock
Mamba-Transformer hybrid: SSM → Attention (replaces SSM → FFN)
```
h = x + Mamba(Norm(x))           # SSM compresses local temporal patterns
out = h + Attention(Norm(h))      # Attention performs global spectral verification
```
- Uses HydraQuasiseparableMixer for the Mamba part
- Uses standard multi-head self-attention for the Attention part
- This "Compress-then-Recall" pattern catches global inconsistencies that pure SSMs miss

### 4. PNMamerStage
One stage of the temporal backbone:
- n_hydra Hydra blocks (default 3) for local temporal modeling
- 1 MamerBlock for global consistency check (only if config.use_mamer)
- If use_mamer=False: all blocks are just Hydra mixers with FFN

Input/Output: (B, T, D) → (B, T, D) for all components.

Test:
```python
import torch
from src.models.hydra_mixer import HydraQuasiseparableMixer, MamerBlock, PNMamerStage
x = torch.randn(2, 32, 128)
mixer = HydraQuasiseparableMixer(d_model=128, d_state=16)
assert mixer(x).shape == (2, 32, 128)
mamer = MamerBlock(d_model=128, d_state=16, n_heads=4)
assert mamer(x).shape == (2, 32, 128)
stage = PNMamerStage(d_model=128, d_state=16, n_hydra=3, n_heads=4, use_mamer=True)
assert stage(x).shape == (2, 32, 128)
```
```

---

## PROMPT 5: W-SS Module + Full V8.0 Backbone

```
Create two final model files:

## 1. `src/models/wss_module.py` — Wavelet-Selective Scan

Dual-stream frequency-native processing that replaces V7.3's naive spatial patch flattening of DWT sub-bands.

### Stream A: TemporalFlickerStream
- Input: HLL sub-band features (B, T, D_hll)
- Spatial pool → 1D temporal scan using bidirectional GRU (lightweight)
- Answers: "Does the flicker follow a drifting path (forgery) or is it random noise (compression)?"
- Output: (B, T, D_out)

### Stream B: SpatialTextureStream
- Input: HHH + HLH sub-band features concatenated (B, T, D_spatial)
- Channel attention gate: learns which frequency bands to trust based on compression profile
- Compression destroys different bands differently (H.264 QP=40 destroys HHH, preserves HLH)
- Scanner learns to ignore destroyed bands dynamically
- Output: (B, T, D_out)

### Gated Cross-Merge Fusion
```python
F_fused = F_spatial * sigmoid(F_temporal)
```
Spatial texture noise only matters if concurrent temporal flicker exists → eliminates false positives from natural textures (leaves, water, hair).

If config.use_wss=False: module is a simple pass-through (returns input unchanged).

## 2. `src/models/backbone_v8.py` — Unified V8.0 Backbone

This is the MAIN model class that assembles everything and switches behavior based on config flags.

```python
class STFMambaV8(nn.Module):
    """Unified backbone: behaves as V7.3 or V8.0 based on config flags."""
    
    def __init__(self, config: STFV8Config):
        # Phase 1: V7.3 stem + spatial stages (always used)
        # Import from backbone_v73.py — reuse the Stem, Stage1, Stage2 with DCConv
        
        # Phase 2: Frequency 
        # 3D-DWT module (always used)
        # WaveletSelectiveScan if config.use_wss else simple HLL gating from V7.3
        
        # Phase 3: Temporal
        # If config.use_hydra: use PNMamerStage blocks
        # Else: use V7.3's MambaVisionMixer blocks
        
        # Phase 4: Classification head
        # Global pool → Linear → (B, 2)
    
    def forward(self, x):
        # x: (B, 3, T, H, W)
        # Returns: {'logits': (B,2), 'hll_energy': (B,), 'all_band_energy': (B,)}
```

The key design: this model MUST be able to load Phase A checkpoints when running Phase B. So:
- V7.3 components (stem, stages 1-2, DWT) keep the SAME parameter names
- V8.0 components (Hydra, Mamer, W-SS) are new parameters that initialize from scratch
- Use a `load_checkpoint_flexible()` function that loads matching keys and skips mismatched ones

Test:
```python
import torch
from src.models.config import STFV8Config
from src.models.backbone_v8 import STFMambaV8

# V7.3 mode
cfg = STFV8Config.debug()
model = STFMambaV8(cfg)
x = torch.randn(2, 3, cfg.num_frames, cfg.img_size, cfg.img_size)
out = model(x)
assert out['logits'].shape == (2, 2)
print(f"V7.3 params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Phase B mode
cfg_b = STFV8Config.debug()
cfg_b.use_hydra = True
cfg_b.use_mamer = True
cfg_b.use_wss = True
model_b = STFMambaV8(cfg_b)
out_b = model_b(x)
assert out_b['logits'].shape == (2, 2)
print(f"V8.0 params: {sum(p.numel() for p in model_b.parameters())/1e6:.2f}M")
```
```

---

## PROMPT 6A: Training Script

```
Create the training infrastructure. We need 4 files:

## 1. `src/training/budget_tracker.py`
```python
class BudgetTracker:
    """Monitors GPU cost and stops training before exceeding budget."""
    def __init__(self, cost_per_hour, max_budget, safety_margin=0.10):
        # Track start_time, cost_per_hour, max_budget
    
    def check(self):
        # Returns dict: spent, remaining, safe_hours_left, should_stop (bool)
    
    def estimate_remaining_epochs(self, avg_epoch_time):
        # Given average epoch duration, estimate how many more we can afford
```

## 2. `src/utils/checkpoint.py`
```python
def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_auc, config):
    """Save training state."""

def load_checkpoint_flexible(path, model, optimizer=None, scheduler=None, scaler=None):
    """Load checkpoint with flexible key matching.
    
    This is CRITICAL for Phase A → Phase B transition:
    - Phase A saves V7.3 architecture weights
    - Phase B loads into V8.0 architecture (has new modules)
    - Matching keys transfer, new keys initialize from scratch
    - Reports which keys were loaded vs skipped
    """
```

## 3. `src/utils/logging.py`
```python
class CSVLogger:
    """Log training metrics to CSV file."""
    COLS = ['epoch', 'phase', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
            'val_auc', 'val_f1', 'lr', 'epoch_time', 'gpu_mem_gb', 'best_auc',
            'cost_usd', 'ce_loss', 'hll_loss', 'timestamp']

class ConsoleLogger:
    """YOLO-style colorful console output with progress bars."""
    # Display per-epoch summary with train/val metrics, cost tracking, ETA
```

## 4. `src/training/train.py` — Main training script

This is adapted from V7.3's train_final.py. Key structure:

```python
"""
STF-Mamba V8.0 Training Script
================================
USAGE:
  python -m src.training.train --config configs/v8_a100.yaml --phase a
  python -m src.training.train --config configs/v8_a100.yaml --phase b --resume outputs/checkpoints/best.pth
"""

def build_model(config):
    """Build STFMambaV8 model from config."""
    from src.models.backbone_v8 import STFMambaV8
    return STFMambaV8(config).cuda()

def build_dataloaders(config, phase):
    """Build train + val dataloaders."""
    from src.data.video_sbi_dataset import VideoSBIDataset
    train_ds = VideoSBIDataset('train', config)
    val_ds = VideoSBIDataset('val', config)
    # DataLoader with pin_memory, prefetch, persistent_workers, drop_last for train

def build_optimizer(model, config):
    """AdamW with differential learning rates.
    
    Backbone (stem + early stages): lr × 0.01
    Temporal modules (Mamba/Hydra): lr × 1.0
    Classification head: lr × 1.0
    """

def build_scheduler(optimizer, config, total_steps, warmup_steps):
    """Cosine schedule with warmup."""
    # Warmup: linear ramp 0 → lr over warmup_epochs
    # Main: cosine decay to min_lr

def train_epoch(model, loader, optimizer, scheduler, scaler, loss_fn, config):
    """One training epoch with gradient accumulation and AMP."""
    # Same pattern as V7.3 train_final.py
    # Use STFV8Loss which handles contrastive HLL
    # Log CE and HLL losses separately

def validate(model, loader, loss_fn, config):
    """Validation with AUC and F1 computation."""
    # Same pattern as V7.3

def main():
    parser: --config, --phase (a/b), --resume (checkpoint path), --epochs (override),
            --batch_size (override), --dry-run (2 epochs on tiny data for testing)
    
    # Load config from YAML, override with CLI args
    # Set feature flags based on --phase
    # Build model, dataloaders, optimizer, scheduler, loss
    # If --resume: load_checkpoint_flexible
    # BudgetTracker monitors cost
    # Training loop:
    #   for epoch in range(start, end):
    #     train_epoch()
    #     validate()
    #     save checkpoint (latest always, best if improved)
    #     budget_tracker.check() → stop if budget exceeded
    #     console output with metrics + cost
```

The training script must handle:
- Resuming from any checkpoint (same architecture or cross-architecture via flexible loading)
- Budget-aware stopping (warns at 80% budget, stops at 90%)
- Gradient accumulation (effective batch = batch_size × grad_accum)
- Mixed precision (bfloat16 on A100)
- Periodic GPU cache clearing (every 15 batches)
```

---

## PROMPT 6B: Config YAML + Setup Scripts

```
Create the deployment files:

## 1. `configs/v8_a100.yaml`
```yaml
data:
  data_root: /runpod-volume/data
  batch_size: 16
  num_workers: 8
  image_size: 224
  num_frames: 32

model:
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
  mlp_ratio: 2.0
  dropout: 0.3
  drop_path: 0.1
  dcconv_kernel_length: 9

training:
  lr: 5.0e-4
  weight_decay: 0.05
  warmup_epochs: 5
  min_lr: 1.0e-6
  label_smoothing: 0.1
  hll_loss_weight: 0.1
  hll_loss_margin: 1.0
  max_grad_norm: 1.0
  gradient_accumulation_steps: 2
  amp_dtype: bfloat16
  cost_per_hour: 1.39
  max_budget: 5.0

phase_a:
  num_epochs: 70
  use_contrastive_hll: true
  use_compression_aug: true
  use_multiscale_drift: true
  use_hll_normalization: true
  use_hydra: false
  use_mamer: false
  use_wss: false

phase_b:
  num_epochs: 50
  use_contrastive_hll: true
  use_compression_aug: true
  use_multiscale_drift: true
  use_hll_normalization: true
  use_hydra: true
  use_mamer: true
  use_wss: true
  lr: 1.0e-4
```

## 2. `scripts/setup_pod.sh`
RunPod pod setup script. Assumes:
- PyTorch 2.1+ template (torch already installed)
- Network volume mounted at /runpod-volume with preprocessed FF++ data
- Repo cloned to /workspace/STF-Mamba_V8.0

Script should:
1. Detect GPU (nvidia-smi) and print info
2. Install Python deps: `pip install -r requirements.txt`
3. Try to install mamba_ssm: `pip install mamba_ssm` (if fails, print warning that Conv1d fallback will be used)
4. Install ffmpeg: `apt-get install -y ffmpeg` (for compression augmentation)
5. Verify data exists at /runpod-volume/data/FaceForensics++/
6. Create symlink: `ln -sf /runpod-volume/data /workspace/STF-Mamba_V8.0/data`
7. Run verify_data.py
8. Print cost estimate and ready message

## 3. `scripts/verify_data.py`
Quick data verification:
- Check frames/ directory exists and count video folders
- Check landmarks/ directory exists and count .npy files
- Check retina/ directory exists and count .npy files
- Check split files (train.json, val.json, test.json) exist
- Load one sample video: read first frame, load landmark, load retina bbox
- Print summary: "N videos, M total frames, ready to train"
- Exit with error code if anything is missing

## 4. `scripts/preprocess.py`
Optional — only needed if preprocessing from scratch:
- Downloads dlib 81-point landmark model
- Runs face detection + landmark extraction on FF++ videos
- Same logic as crop_dlib_ff.py and crop_retina_ff.py from SBI reference
- Saves to the standard frames/ + landmarks/ + retina/ format

## 5. `README.md`
Project README with:
- One-paragraph description
- Quick start (3 commands: clone, setup, train)
- Project structure
- Phase A vs Phase B explanation
- Budget estimates
- Link to paper PDF
```

---

## PROMPT 6C: Evaluation Script

```
Create `src/training/evaluate.py` — Cross-dataset evaluation on REAL deepfakes.

This is the CRITICAL test: V7.3 got 98.57% on SBI validation but 51.55% on real deepfakes. This script evaluates on real FF++ manipulation methods.

## Evaluation protocol (matching SBI paper):
For each test video:
1. Extract N frames (already preprocessed in frames/ directory)
2. Run model on N frames as a video clip
3. Video-level prediction = model output (it's already video-level)
4. Compute AUC, accuracy, precision, recall, F1 at video level

## Data for evaluation:
Real videos: {data_root}/FaceForensics++/original_sequences/youtube/raw/frames/
Fake videos: {data_root}/FaceForensics++/manipulated_sequences/{method}/raw/frames/
  where method ∈ {Deepfakes, Face2Face, FaceSwap, NeuralTextures}

Split: use test.json to filter which videos to evaluate on

## Usage:
```bash
python -m src.training.evaluate \
    --checkpoint outputs/checkpoints/best.pth \
    --config configs/v8_a100.yaml \
    --methods Deepfakes Face2Face FaceSwap NeuralTextures \
    --output outputs/eval/
```

## Output:
- Per-method metrics: AUC, accuracy, precision, recall, F1
- Overall metrics across all methods
- Save results as JSON
- Print comparison table to console:
```
Method          | N    | AUC    | Acc    | Interpretation
Deepfakes       | 140  | 85.3%  | 78.6%  | Working
Face2Face       | 140  | 72.1%  | 68.2%  | Moderate
FaceSwap        | 140  | 79.5%  | 74.3%  | Working
NeuralTextures  | 140  | 65.8%  | 61.4%  | Needs work
Overall         | 700  | 75.7%  | 70.6%  | BIG improvement from 51%
```

## SBI validation mode:
Also support --sbi-test flag that evaluates on SBI validation set (to verify SBI performance didn't degrade).

## Important: The evaluation dataset class is DIFFERENT from training:
- NO SBI generation (we're testing on real fakes, not self-blended)
- Load real frames from original_sequences as real, manipulated_sequences as fake
- Standard face crop + resize, no augmentation
- Video-level: load 32 frames per video, run through model as one clip
```

---

## PROMPT 7: Integration Test + Git Push

```
Now let's do a final integration test to verify everything connects properly before pushing to GitHub.

Create `tests/test_integration.py`:

```python
"""Integration test — verifies full pipeline on CPU with dummy data."""
import torch
import numpy as np
import tempfile
import os

def test_config():
    from src.models.config import STFV8Config
    cfg_a = STFV8Config.phase_a()
    cfg_b = STFV8Config.phase_b()
    cfg_d = STFV8Config.debug()
    print(f"Phase A: {cfg_a.describe()}")
    print(f"Phase B: {cfg_b.describe()}")
    print(f"Debug: {cfg_d.describe()}")
    assert cfg_a.use_contrastive_hll == True
    assert cfg_a.use_hydra == False
    assert cfg_b.use_hydra == True
    print("✓ Config OK")

def test_losses():
    from src.models.config import STFV8Config
    from src.models.losses import STFV8Loss
    cfg = STFV8Config.phase_a()
    loss_fn = STFV8Loss(cfg)
    out = {
        'logits': torch.randn(8, 2),
        'hll_energy': torch.rand(8),
        'all_band_energy': torch.ones(8),
    }
    labels = torch.tensor([0,0,0,0,1,1,1,1])
    losses = loss_fn(out, labels)
    losses['total'].backward()
    print(f"✓ Loss OK — total={losses['total']:.4f}, ce={losses['ce']:.4f}, hll={losses['hll']:.4f}")

def test_backbone_v73():
    from src.models.config import STFV8Config
    from src.models.backbone_v73 import STFMambaV73
    cfg = STFV8Config.debug()
    model = STFMambaV73(cfg)
    x = torch.randn(2, 3, cfg.num_frames, cfg.img_size, cfg.img_size)
    out = model(x)
    assert out['logits'].shape == (2, 2)
    n = sum(p.numel() for p in model.parameters())
    print(f"✓ V7.3 Backbone OK — {n/1e6:.2f}M params, logits={out['logits'].shape}")

def test_hydra():
    from src.models.hydra_mixer import HydraQuasiseparableMixer, MamerBlock, PNMamerStage
    x = torch.randn(2, 16, 64)
    mixer = HydraQuasiseparableMixer(d_model=64, d_state=8)
    assert mixer(x).shape == x.shape
    mamer = MamerBlock(d_model=64, d_state=8, n_heads=4)
    assert mamer(x).shape == x.shape
    stage = PNMamerStage(d_model=64, d_state=8, n_hydra=2, n_heads=4, use_mamer=True)
    assert stage(x).shape == x.shape
    print("✓ Hydra + Mamer OK")

def test_backbone_v8():
    from src.models.config import STFV8Config
    from src.models.backbone_v8 import STFMambaV8
    # V7.3 mode
    cfg = STFV8Config.debug()
    model = STFMambaV8(cfg)
    x = torch.randn(2, 3, cfg.num_frames, cfg.img_size, cfg.img_size)
    out = model(x)
    assert out['logits'].shape == (2, 2)
    n73 = sum(p.numel() for p in model.parameters())
    # V8.0 mode
    cfg.use_hydra = True; cfg.use_mamer = True; cfg.use_wss = True
    model_v8 = STFMambaV8(cfg)
    out_v8 = model_v8(x)
    assert out_v8['logits'].shape == (2, 2)
    n80 = sum(p.numel() for p in model_v8.parameters())
    print(f"✓ V8.0 Backbone OK — V7.3={n73/1e6:.2f}M, V8.0={n80/1e6:.2f}M")

def test_augmentations():
    from src.data.augmentations import MultiScaleBrownianDrift, CompressionAugmentation
    mask = np.zeros((224, 224), dtype=np.float32)
    mask[50:180, 50:180] = 1.0
    drift = MultiScaleBrownianDrift(step_range=(0.5, 5.0))
    masks = drift.apply_drift(mask, 32)
    assert len(masks) == 32
    assert all(m.shape == (224, 224) for m in masks)
    print("✓ Augmentations OK")

if __name__ == '__main__':
    test_config()
    test_losses()
    test_augmentations()
    test_backbone_v73()
    test_hydra()
    test_backbone_v8()
    print("\n✅ ALL INTEGRATION TESTS PASSED — ready to push to GitHub")
```

Run this test. If it passes, do:
```bash
git add -A
git commit -m "STF-Mamba V8.0: complete implementation with feature flags"
git push origin main
```

Then the project is ready for RunPod deployment.
```

---

## RunPod Execution Prompts (after code is ready)

These are NOT Cursor prompts — these are terminal commands for RunPod:

### RunPod Step 1: Setup
```bash
cd /workspace
git clone https://github.com/AbdelRahman-Madboly/STF-Mamba_V8.0.git
cd STF-Mamba_V8.0
bash scripts/setup_pod.sh
```

### RunPod Step 2: Phase A Training
```bash
python -m src.training.train --config configs/v8_a100.yaml --phase a
```

### RunPod Step 3: Evaluate Phase A
```bash
python -m src.training.evaluate \
    --checkpoint outputs/checkpoints/best.pth \
    --config configs/v8_a100.yaml \
    --methods Deepfakes Face2Face FaceSwap NeuralTextures
```

### RunPod Step 4: Phase B (if budget remains + Phase A AUC > 70%)
```bash
python -m src.training.train \
    --config configs/v8_a100.yaml \
    --phase b \
    --resume outputs/checkpoints/best.pth
```
