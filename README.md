# STF-Mamba V8.0 — Implementation Guide

## Budget Plan: $8 RunPod ($5 Training + $3 Reserve)

### What You Have
| Item | Amount |
|---|---|
| RunPod balance | $8.00 |
| Training budget | $5.00 |
| Debug/eval reserve | $3.00 |

### Recommended GPU: RTX 4090 ($0.39/hr)
$5.00 ÷ $0.39/hr = **12.8 hours** of GPU time → enough for ~145 epochs

---

## Step-by-Step Execution Plan

### Step 0: Preparation (before starting RunPod)
- [ ] Upload these files to your GitHub repo or Google Drive
- [ ] Copy your V7.3 `preprocess_data.py` and `train.py` (for the dataloader code)
- [ ] Make sure your preprocessed FF++ dataset is accessible (RunPod volume or download script)

### Step 1: Start RunPod ($0.00)
1. Create a **RTX 4090** pod with PyTorch template
2. SSH in and clone your project:
   ```bash
   cd /workspace
   git clone <your-repo> stf_mamba_v8
   # OR upload via SCP/Drive
   ```

### Step 2: Setup Environment (~5 min, ~$0.03)
```bash
cd /workspace/stf_mamba_v8
bash setup_pod.sh
```

### Step 3: CRITICAL — Connect Your Dataloader
The `train_v8.py` has a placeholder `build_dataloaders()`. You MUST replace it
with your actual Video-SBI pipeline from V7.3's `train.py`. The key changes:

```python
# In your dataloader, ADD these augmentations:

# 1. Compression augmentation (new)
import subprocess
def compress_video(frames, qp=None):
    """Random H.264 re-compression to simulate different codecs."""
    if qp is None:
        qp = random.randint(23, 40)  # Random quality
    # Apply ffmpeg compression or use torchvision.io
    # This is the #1 most important augmentation fix
    ...

# 2. Multi-scale drift (replace fixed ±2px)
drift_scale = random.uniform(0.5, 5.0)  # Was: fixed 2.0

# 3. img_size = 224 (not 256) — matches ConvNeXt V2 pretraining
```

### Step 4: Phase A Training (~$2.50, ~6.4 hours)
```bash
python train_v8.py --phase a --budget 2.50 --gpu "RTX 4090"
```

**What Phase A does:**
- V7.3 architecture (ConvNeXt V2 + Bi-Mamba + FFN)
- NEW: Contrastive HLL Loss (replaces MSE-to-zero)
- NEW: HLL energy normalization (compression invariance)
- NEW: Multi-scale Brownian drift + compression augmentation

**Expected results:** 70-85% AUC on real deepfakes (up from 51%)

### Step 5: Evaluate Phase A (~$0.20)
```bash
python evaluate_real_deepfakes.py --checkpoint /workspace/checkpoints/phase_a_best.pt
```

**Decision point:**
- AUC > 70% → proceed to Phase B
- AUC < 60% → debug augmentation pipeline (check if compression aug is working)
- AUC 60-70% → consider running more Phase A epochs before Phase B

### Step 6: Phase B Training (~$2.50, ~6.4 hours)
```bash
python train_v8.py --phase b --budget 2.50 --gpu "RTX 4090" \
  --resume /workspace/checkpoints/phase_a_best.pt
```

**What Phase B adds:**
- Hydra Quasiseparable Mixer (replaces Bi-Mamba)
- Mamer Attention (replaces FFN for global spectral recall)
- W-SS dual-stream scanning (frequency-native processing)
- Loads Phase A weights (compatible keys transfer automatically)

**Expected results:** 85-95% AUC on cross-dataset evaluation

### Step 7: Final Evaluation (~$0.30)
```bash
# Cross-dataset evaluation
python evaluate_real_deepfakes.py --checkpoint /workspace/checkpoints/phase_b_best.pt
```

---

## File Structure

```
stf_mamba_v8/
├── config_v8.py              # Config with feature flags
├── train_v8.py               # Training script with budget tracking
├── setup_pod.sh              # RunPod environment setup
├── README.md                 # This file
└── modules/
    ├── __init__.py
    ├── backbone_v8.py        # Main model (ConvNeXt V2 + DWT + temporal)
    ├── hydra_mixer.py        # Hydra + Mamer + PNMamer stages
    ├── wss_module.py         # Wavelet-Selective Scan
    └── losses.py             # Contrastive HLL Loss
```

## What You Still Need To Do

### Must Do (before training)
1. **Connect your Video-SBI dataloader** — copy from V7.3's `train.py`
2. **Add compression augmentation** — random H.264 re-encoding (QP 23-40)
3. **Change img_size to 224** in your preprocessing pipeline
4. **Verify dataset paths** match RunPod volume mount

### Nice To Have (after Phase A works)
1. Thin-plate spline (TPS) mask deformation for diverse boundary drift
2. Variable frame-rate simulation (drop/duplicate frames randomly)
3. Multiple blending methods (alpha, Poisson, Laplacian pyramid)

## Development Workflow

### Use Claude Opus 4.6 (here) for:
- Architectural decisions and debugging strategies
- Complex module design and mathematical reasoning
- Reviewing evaluation results and proposing next steps

### Use Cursor for:
- Interactive debugging on RunPod (SSH terminal)
- Quick syntax fixes and error resolution
- Iterating on dataloader/augmentation code
- Running experiments and checking outputs

### Recommended Cursor setup:
1. Open the stf_mamba_v8 folder in Cursor
2. Set Claude as the AI model in Cursor settings
3. Use Cursor's inline chat for quick fixes
4. Use Claude.ai (here) for deep architectural discussions
