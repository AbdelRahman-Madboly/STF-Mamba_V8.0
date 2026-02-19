#!/bin/bash
# ============================================
# STF-Mamba V8.0 - RunPod Setup Script
# ============================================
# Usage: bash setup_pod.sh
# GPU: RTX 4090 (recommended) or A100
# Budget: $5 total ($2.50 Phase A + $2.50 Phase B)
# ============================================

set -e

echo "================================================"
echo "  STF-Mamba V8.0 - RunPod Environment Setup"
echo "================================================"

# === 1. System packages ===
echo "[1/6] Installing system packages..."
apt-get update -qq && apt-get install -y -qq \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx git wget unzip \
    > /dev/null 2>&1
echo "  ✓ System packages installed"

# === 2. Python packages ===
echo "[2/6] Installing Python packages..."
pip install --quiet --upgrade pip

# Core ML
pip install --quiet \
    torch torchvision torchaudio \
    timm>=0.9.7 \
    einops>=0.7.0 \
    transformers>=4.35.0

# Mamba SSM (try to install, fallback gracefully)
echo "  Installing mamba_ssm..."
pip install --quiet causal-conv1d 2>/dev/null || echo "  ⚠ causal-conv1d not available, using fallback"
pip install --quiet mamba-ssm 2>/dev/null || echo "  ⚠ mamba_ssm not available, using Conv1d fallback"

# Data & augmentation
pip install --quiet \
    opencv-python-headless \
    albumentations \
    decord \
    dlib \
    face-recognition \
    Pillow \
    imageio imageio-ffmpeg

# Science & metrics
pip install --quiet \
    numpy "numpy<2.0" \
    scipy scikit-learn scikit-image \
    pandas matplotlib seaborn \
    torchmetrics

# Monitoring
pip install --quiet tensorboard tqdm colorama termcolor pyyaml

echo "  ✓ Python packages installed"

# === 3. Verify GPU ===
echo "[3/6] Verifying GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'  ✓ GPU: {gpu} ({vram:.1f} GB VRAM)')
else:
    print('  ✗ NO GPU DETECTED')
    exit(1)
"

# === 4. Verify key imports ===
echo "[4/6] Verifying imports..."
python3 -c "
import timm
import torch
try:
    from mamba_ssm import Mamba
    print('  ✓ mamba_ssm available (optimized SSM)')
except:
    print('  ⚠ mamba_ssm not available (using Conv1d fallback)')

# Test ConvNeXt V2 
model = timm.create_model('convnextv2_base', pretrained=False, num_classes=0)
print(f'  ✓ ConvNeXt V2-Base: {sum(p.numel() for p in model.parameters()):,} params')
print('  ✓ All imports verified')
"

# === 5. Download ConvNeXt V2 weights (cache for faster startup) ===
echo "[5/6] Pre-caching ConvNeXt V2-Base weights..."
python3 -c "
import timm
model = timm.create_model('convnextv2_base', pretrained=True, num_classes=0)
print('  ✓ ConvNeXt V2-Base weights cached')
" 2>/dev/null || echo "  ⚠ Weight download failed (will retry during training)"

# === 6. Verify project structure ===
echo "[6/6] Checking project files..."
REQUIRED_FILES=(
    "config_v8.py"
    "train_v8.py"
    "modules/__init__.py"
    "modules/backbone_v8.py"
    "modules/hydra_mixer.py"
    "modules/wss_module.py"
    "modules/losses.py"
)

MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "/workspace/stf_mamba_v8/$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ✗ MISSING: $f"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -gt 0 ]; then
    echo "  ⚠ $MISSING files missing. Upload project to /workspace/stf_mamba_v8/"
fi

# === Done ===
echo ""
echo "================================================"
echo "  Setup complete! Next steps:"
echo ""
echo "  Phase A (first \$2.50):"
echo "    cd /workspace/stf_mamba_v8"
echo "    python train_v8.py --phase a --budget 2.50"
echo ""
echo "  Phase B (next \$2.50):"
echo "    python train_v8.py --phase b --budget 2.50 \\"
echo "      --resume /workspace/checkpoints/phase_a_best.pt"
echo "================================================"
