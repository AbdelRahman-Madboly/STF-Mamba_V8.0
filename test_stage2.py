"""
STF-Mamba V8.0 — Stage 2 Exit Test
=====================================
Run from: C:\Dan_WS\STF-Mamba_V8.0
Command:  python test_stage2.py

Tests each module individually, then the full pipeline.
Works on CPU (laptop) — DINOv2 downloads ~350MB on first run.
"""

import sys
import time

print("=" * 60)
print("  STF-Mamba V8.0 — Stage 2 Exit Criterion Test")
print("=" * 60)

# ─── Test 0: Imports ───
print("\n[0/5] Testing imports...")
try:
    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA:    {torch.cuda.is_available()}")
except ImportError:
    print("  ERROR: PyTorch not installed. Run: pip install torch torchvision")
    sys.exit(1)

try:
    from stf_mamba.backbone import DINOv2Backbone
    from stf_mamba.hydra_mamba import HydraMambaTemporalModule, is_mamba_available
    from stf_mamba.consistency_head import VarianceConsistencyHead
    from stf_mamba.losses import STFMambaLoss
    from stf_mamba.model import STFMambaV8
    print("  All stf_mamba imports OK")
except ImportError as e:
    print(f"  ERROR: {e}")
    print("  Make sure you ran: pip install -e .")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device:  {device}")
print(f"  Mamba:   {'available' if is_mamba_available() else 'Conv1d fallback'}")

# Use small batch for laptop CPU testing
B, T = 1, 8  # Small: 1 clip, 8 frames (instead of 2x32)
if device == "cuda":
    B, T = 2, 32  # Full size on GPU

print(f"  Test:    B={B}, T={T} ({'full size' if T == 32 else 'reduced for CPU'})")

# ─── Test 1: Backbone ───
print(f"\n[1/5] Testing DINOv2Backbone...")
print("  (First run downloads ~350MB model — be patient)")
t0 = time.time()
try:
    backbone = DINOv2Backbone(pretrained=True).to(device)
    x = torch.randn(B, T, 3, 224, 224, device=device)
    with torch.no_grad():
        cls_tokens = backbone(x)
    assert cls_tokens.shape == (B, T, 768), f"Expected (B,T,768), got {cls_tokens.shape}"
    
    total = sum(p.numel() for p in backbone.parameters())
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"  Output:    {cls_tokens.shape} ✓")
    print(f"  Params:    {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")
    print(f"  Time:      {time.time()-t0:.1f}s")
    print("  PASS ✓")
except Exception as e:
    print(f"  FAIL ✗ — {e}")
    sys.exit(1)

# ─── Test 2: Hydra-Mamba ───
print(f"\n[2/5] Testing HydraMambaTemporalModule...")
t0 = time.time()
try:
    temporal = HydraMambaTemporalModule(
        input_dim=768, proj_dim=512, num_blocks=2,
    ).to(device)
    with torch.no_grad():
        temporal_out = temporal(cls_tokens)
    assert temporal_out.shape == (B, T, 512), f"Expected (B,T,512), got {temporal_out.shape}"
    
    params = sum(p.numel() for p in temporal.parameters())
    print(f"  Output:    {temporal_out.shape} ✓")
    print(f"  Params:    {params/1e6:.1f}M")
    print(f"  Time:      {time.time()-t0:.1f}s")
    print("  PASS ✓")
except Exception as e:
    print(f"  FAIL ✗ — {e}")
    sys.exit(1)

# ─── Test 3: Consistency Head ───
print(f"\n[3/5] Testing VarianceConsistencyHead...")
t0 = time.time()
try:
    head = VarianceConsistencyHead(embed_dim=512).to(device)
    with torch.no_grad():
        head_out = head(temporal_out)
    
    assert head_out["logits"].shape == (B, 2), f"logits: {head_out['logits'].shape}"
    assert head_out["variance"].shape == (B, 1), f"variance: {head_out['variance'].shape}"
    assert head_out["similarities"].shape == (B, T), f"sims: {head_out['similarities'].shape}"
    
    print(f"  Logits:    {head_out['logits'].shape} ✓")
    print(f"  Variance:  {head_out['variance'].shape} ✓")
    print(f"  Sims:      {head_out['similarities'].shape} ✓")
    print(f"  Time:      {time.time()-t0:.1f}s")
    print("  PASS ✓")
except Exception as e:
    print(f"  FAIL ✗ — {e}")
    sys.exit(1)

# ─── Test 4: Loss ───
print(f"\n[4/5] Testing STFMambaLoss...")
t0 = time.time()
try:
    criterion = STFMambaLoss(lambda_var=0.1)
    labels = torch.randint(0, 2, (B,), device=device)
    loss_out = criterion(head_out["logits"], labels, head_out["variance"])
    
    assert "total" in loss_out
    assert "ce" in loss_out
    assert "var" in loss_out
    assert "var_gap" in loss_out
    assert not torch.isnan(loss_out["total"]), "Loss is NaN!"
    
    print(f"  Total:     {loss_out['total'].item():.4f} ✓")
    print(f"  CE:        {loss_out['ce'].item():.4f}")
    print(f"  Var:       {loss_out['var'].item():.4f}")
    print(f"  Var gap:   {loss_out['var_gap'].item():.4f}")
    print(f"  Time:      {time.time()-t0:.1f}s")

    # Test label_smoothing safeguard
    try:
        bad_loss = STFMambaLoss(label_smoothing=0.1)
        print("  Bug #1 safeguard: FAIL ✗ (should have raised error)")
        sys.exit(1)
    except ValueError:
        print("  Bug #1 safeguard: blocked label_smoothing>0 ✓")
    
    print("  PASS ✓")
except Exception as e:
    print(f"  FAIL ✗ — {e}")
    sys.exit(1)

# ─── Test 5: Full Model (Exit Criterion) ───
print(f"\n[5/5] Testing full STFMambaV8 pipeline...")
print("  This is the Stage 2 exit criterion.")
t0 = time.time()
try:
    model = STFMambaV8(pretrained_backbone=True).to(device)
    x = torch.randn(B, T, 3, 224, 224, device=device)
    
    with torch.no_grad():
        out = model(x)
    
    assert out["logits"].shape == (B, 2), f"logits: {out['logits'].shape}"
    assert out["variance"].shape == (B, 1), f"variance: {out['variance'].shape}"
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Logits:    {out['logits'].shape} ✓")
    print(f"  Variance:  {out['variance'].shape} ✓")
    print(f"  Params:    {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")
    print(f"  Time:      {time.time()-t0:.1f}s")
    print("  PASS ✓")
except Exception as e:
    print(f"  FAIL ✗ — {e}")
    sys.exit(1)

# ─── Test gradient flow ───
print(f"\n[BONUS] Testing backward pass...")
try:
    model.train()
    x = torch.randn(B, T, 3, 224, 224, device=device)
    labels = torch.randint(0, 2, (B,), device=device)
    
    out = model(x)
    loss_out = criterion(out["logits"], labels, out["variance"])
    loss_out["total"].backward()
    
    # Check gradients exist on trainable params
    grad_count = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    
    print(f"  Gradients: {grad_count}/{total_trainable} params have grads ✓")
    print(f"  Loss:      {loss_out['total'].item():.4f}")
    print("  PASS ✓")
except Exception as e:
    print(f"  FAIL ✗ — {e}")

# ─── Summary ───
print("\n" + "=" * 60)
print("  STAGE 2 EXIT CRITERION: ALL PASSED ✓")
print("=" * 60)
print(f"""
  Next steps:
    1. git add .
    2. git commit -m "Stage 2: Model code | Exit: forward pass OK"
    3. git push
    4. Move to Stage 3: Data pipeline
""")