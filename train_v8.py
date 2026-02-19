#!/usr/bin/env python3
"""
STF-Mamba V8.0 - Training Script
===================================

Budget-aware training for RunPod deployment.

Usage:
    # Phase A: V7.3 + fixes (first $2.50)
    python train_v8.py --phase a --budget 2.50 --gpu rtx4090
    
    # Phase B: Full V8.0 (next $2.50, loads Phase A checkpoint)
    python train_v8.py --phase b --budget 2.50 --gpu rtx4090 --resume checkpoints/phase_a_best.pt
    
    # Debug run (CPU)
    python train_v8.py --phase debug
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_v8 import STFV8Config
from modules.backbone_v8 import STFMambaV8
from modules.losses import STFV8Loss


def parse_args():
    parser = argparse.ArgumentParser(description='STF-Mamba V8.0 Training')
    parser.add_argument('--phase', type=str, default='a', choices=['a', 'b', 'debug'],
                        help='Training phase: a (V7.3+fixes), b (full V8.0), debug')
    parser.add_argument('--budget', type=float, default=2.50,
                        help='GPU budget in USD')
    parser.add_argument('--gpu', type=str, default='RTX 4090',
                        help='GPU type for cost estimation')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--data_root', type=str, default='/workspace/data',
                        help='Dataset root directory')
    parser.add_argument('--output_dir', type=str, default='/workspace/outputs',
                        help='Output directory')
    
    # Override any config param
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--img_size', type=int, default=None)
    parser.add_argument('--num_frames', type=int, default=None)
    
    return parser.parse_args()


def build_config(args) -> STFV8Config:
    """Build config from args."""
    if args.phase == 'debug':
        config = STFV8Config.debug()
    elif args.phase == 'b':
        config = STFV8Config.phase_b()
    else:
        config = STFV8Config.phase_a()
    
    # Apply overrides
    if args.epochs: config.max_epochs = args.epochs
    if args.batch_size: config.batch_size = args.batch_size
    if args.lr: config.lr = args.lr
    if args.img_size: config.img_size = args.img_size
    if args.num_frames: config.num_frames = args.num_frames
    if args.data_root: config.data_root = args.data_root
    if args.output_dir: config.output_dir = args.output_dir
    
    return config


class BudgetTracker:
    """Track GPU cost and stop before exceeding budget."""
    
    GPU_COSTS = {
        'RTX 4090': 0.39,
        'RTX 3090': 0.22,
        'A40': 0.79,
        'A100 80GB': 1.64,
        'A100 40GB': 1.24,
        'H100': 2.49,
    }
    
    def __init__(self, budget_usd: float, gpu_name: str = 'RTX 4090'):
        self.budget = budget_usd
        self.cost_per_hour = self.GPU_COSTS.get(gpu_name, 0.39)
        self.start_time = time.time()
        self.gpu_name = gpu_name
        
    def elapsed_hours(self) -> float:
        return (time.time() - self.start_time) / 3600
    
    def cost_so_far(self) -> float:
        return self.elapsed_hours() * self.cost_per_hour
    
    def budget_remaining(self) -> float:
        return self.budget - self.cost_so_far()
    
    def can_afford_epoch(self, sec_per_epoch: float) -> bool:
        """Check if we can afford another epoch."""
        epoch_cost = (sec_per_epoch / 3600) * self.cost_per_hour
        return self.budget_remaining() > epoch_cost * 1.1  # 10% safety margin
    
    def status(self) -> str:
        elapsed = self.elapsed_hours()
        cost = self.cost_so_far()
        remaining = self.budget_remaining()
        return (f"[Budget] ${cost:.3f} / ${self.budget:.2f} spent | "
                f"${remaining:.3f} remaining | {elapsed:.2f}h on {self.gpu_name}")


class EpochTimer:
    """Track epoch timing for budget estimation."""
    
    def __init__(self):
        self.times = []
        
    def start(self):
        self._start = time.time()
        
    def stop(self):
        elapsed = time.time() - self._start
        self.times.append(elapsed)
        return elapsed
    
    def avg_seconds(self) -> float:
        return sum(self.times) / len(self.times) if self.times else 300  # Default 5min


def build_dataloaders(config: STFV8Config):
    """
    Build training and validation dataloaders.
    
    This is a PLACEHOLDER — you'll replace this with your actual 
    Video-SBI dataloader from V7.3's train.py.
    
    Key changes from V7.3:
    1. Add compression augmentation (random H.264 re-encoding)
    2. Multi-scale Brownian drift (not just ±2px)
    3. img_size=224 (not 256) to match ConvNeXt pretraining
    """
    from torch.utils.data import TensorDataset
    
    logger.warning("=" * 60)
    logger.warning("USING PLACEHOLDER DATALOADERS")
    logger.warning("Replace build_dataloaders() with your Video-SBI pipeline")
    logger.warning("=" * 60)
    
    B = config.batch_size
    T = config.num_frames
    H = W = config.img_size
    
    # Placeholder random data
    train_data = TensorDataset(
        torch.randn(64, 3, T, H, W),
        torch.randint(0, 2, (64,)),
    )
    val_data = TensorDataset(
        torch.randn(16, 3, T, H, W),
        torch.randint(0, 2, (16,)),
    )
    
    train_loader = DataLoader(
        train_data, batch_size=B, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=B, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
    )
    
    return train_loader, val_loader


def train_one_epoch(
    model, train_loader, criterion, optimizer, scaler, 
    config, device, epoch, grad_accum_steps
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_ce = 0
    total_hll = 0
    total_correct = 0
    total_samples = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (videos, labels) in enumerate(train_loader):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast(enabled=config.use_mixed_precision):
            # Forward pass
            outputs = model(videos)
            
            # Compute loss
            loss_dict = criterion(
                logits=outputs['logits'],
                labels=labels,
                hll_energy=outputs['hll_energy'],
                all_band_energy=outputs.get('all_band_energy'),
            )
            loss = loss_dict['loss'] / grad_accum_steps
        
        # Backward
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Track metrics
        total_loss += loss_dict['loss'].item()
        total_ce += loss_dict['loss_ce'].item()
        total_hll += loss_dict['loss_hll'].item()
        
        preds = outputs['logits'].argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
    
    n_batches = len(train_loader)
    return {
        'train_loss': total_loss / max(n_batches, 1),
        'train_ce': total_ce / max(n_batches, 1),
        'train_hll': total_hll / max(n_batches, 1),
        'train_acc': total_correct / max(total_samples, 1) * 100,
    }


@torch.no_grad()
def validate(model, val_loader, criterion, config, device):
    """Validation loop."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_probs = []
    all_labels = []
    
    for videos, labels in val_loader:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast(enabled=config.use_mixed_precision):
            outputs = model(videos)
            loss_dict = criterion(
                logits=outputs['logits'],
                labels=labels,
                hll_energy=outputs['hll_energy'],
                all_band_energy=outputs.get('all_band_energy'),
            )
        
        total_loss += loss_dict['loss'].item()
        preds = outputs['logits'].argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        
        probs = F.softmax(outputs['logits'], dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Compute AUC
    import numpy as np
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    
    n_batches = len(val_loader)
    return {
        'val_loss': total_loss / max(n_batches, 1),
        'val_acc': total_correct / max(total_samples, 1) * 100,
        'val_auc': auc,
    }


def save_checkpoint(model, optimizer, scaler, epoch, metrics, path):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'metrics': metrics,
    }, path)
    logger.info(f"Saved checkpoint: {path}")


def load_checkpoint(model, optimizer, scaler, path, device):
    """Load checkpoint (supports Phase A → Phase B upgrade)."""
    ckpt = torch.load(path, map_location=device)
    
    # Flexible loading: skip mismatched keys (for Phase A → B)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in ckpt['model_state_dict'].items() 
                       if k in model_dict and v.shape == model_dict[k].shape}
    
    skipped = set(ckpt['model_state_dict'].keys()) - set(pretrained_dict.keys())
    if skipped:
        logger.info(f"Skipped {len(skipped)} keys from checkpoint (architecture mismatch):")
        for k in list(skipped)[:5]:
            logger.info(f"  - {k}")
        if len(skipped) > 5:
            logger.info(f"  ... and {len(skipped) - 5} more")
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    logger.info(f"Loaded {len(pretrained_dict)}/{len(model_dict)} params from {path}")
    logger.info(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    
    # Try to load optimizer/scaler (may fail on architecture change)
    try:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    except:
        logger.warning("Could not load optimizer state (architecture changed). Starting fresh optimizer.")
    
    return ckpt.get('epoch', 0)


def main():
    args = parse_args()
    config = build_config(args)
    
    # Print config
    logger.info("=" * 60)
    logger.info(f"STF-Mamba V8.0 Training")
    logger.info(f"Mode: {config.describe_mode()}")
    logger.info(f"Budget: ${args.budget:.2f} on {args.gpu}")
    cost_est = config.estimate_cost(args.gpu)
    logger.info(f"Estimated: {cost_est}")
    logger.info("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    
    # Budget tracker
    budget = BudgetTracker(args.budget, args.gpu)
    epoch_timer = EpochTimer()
    
    # Build model
    model = STFMambaV8(config).to(device)
    
    # Build optimizer with differential LR
    param_groups = model.get_optimizer_param_groups()
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    
    # Loss
    criterion = STFV8Loss(
        hll_weight=config.hll_loss_weight,
        hll_margin=config.hll_loss_margin,
        hll_temperature=config.hll_loss_temperature,
        hll_mode="margin" if config.use_contrastive_hll else "margin",
        label_smoothing=config.label_smoothing,
        normalize_hll=config.use_hll_normalization,
    )
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=config.use_mixed_precision)
    
    # LR scheduler: cosine with warmup
    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return epoch / max(config.warmup_epochs, 1)
        progress = (epoch - config.warmup_epochs) / max(config.max_epochs - config.warmup_epochs, 1)
        return max(config.min_lr / config.lr, 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Resume
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scaler, args.resume, device)
    
    # Data
    train_loader, val_loader = build_dataloaders(config)
    
    # Output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config.to_dict(), f, indent=2, default=str)
    
    # Training loop
    best_auc = 0.0
    training_log = []
    
    logger.info(f"\nStarting training: {config.max_epochs} epochs")
    logger.info(f"Effective batch size: {config.batch_size * config.grad_accum_steps}")
    
    for epoch in range(start_epoch, config.max_epochs):
        epoch_timer.start()
        
        # Check budget
        if not budget.can_afford_epoch(epoch_timer.avg_seconds()):
            logger.warning(f"\n{'='*60}")
            logger.warning(f"BUDGET LIMIT REACHED at epoch {epoch}")
            logger.warning(budget.status())
            logger.warning(f"{'='*60}")
            break
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            config, device, epoch, config.grad_accum_steps
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, config, device)
        
        # Step scheduler
        scheduler.step()
        
        # Timing
        epoch_time = epoch_timer.stop()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        log_entry = {
            'epoch': epoch + 1,
            'phase': args.phase,
            **train_metrics,
            **val_metrics,
            'lr': current_lr,
            'epoch_time': round(epoch_time, 1),
            'cost_usd': round(budget.cost_so_far(), 3),
        }
        training_log.append(log_entry)
        
        logger.info(
            f"Epoch {epoch+1:3d}/{config.max_epochs} | "
            f"Loss: {train_metrics['train_loss']:.4f} | "
            f"Acc: {train_metrics['train_acc']:.1f}% | "
            f"Val AUC: {val_metrics['val_auc']:.4f} | "
            f"HLL: {train_metrics['train_hll']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.0f}s | "
            f"{budget.status()}"
        )
        
        # Save best
        if val_metrics['val_auc'] > best_auc:
            best_auc = val_metrics['val_auc']
            save_checkpoint(
                model, optimizer, scaler, epoch, val_metrics,
                ckpt_dir / f'phase_{args.phase}_best.pt'
            )
            logger.info(f"  ★ New best AUC: {best_auc:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scaler, epoch, val_metrics,
                ckpt_dir / f'phase_{args.phase}_epoch{epoch+1}.pt'
            )
    
    # Save training log
    with open(output_dir / f'training_log_phase_{args.phase}.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING COMPLETE")
    logger.info(f"Phase: {args.phase.upper()}")
    logger.info(f"Best AUC: {best_auc:.4f}")
    logger.info(f"Epochs completed: {len(training_log)}")
    logger.info(f"{budget.status()}")
    logger.info(f"{'='*60}")
    
    # Print next steps
    if args.phase == 'a':
        logger.info("\n=== NEXT STEPS ===")
        logger.info(f"1. Evaluate Phase A model on real deepfakes:")
        logger.info(f"   python evaluate_real_deepfakes.py --checkpoint {ckpt_dir}/phase_a_best.pt")
        logger.info(f"2. If AUC > 70%, proceed to Phase B:")
        logger.info(f"   python train_v8.py --phase b --resume {ckpt_dir}/phase_a_best.pt --budget 2.50")
        logger.info(f"3. If AUC < 60%, check augmentation pipeline and loss curves")
    elif args.phase == 'b':
        logger.info("\n=== NEXT STEPS ===")
        logger.info(f"1. Full cross-dataset evaluation:")
        logger.info(f"   python evaluate_real_deepfakes.py --checkpoint {ckpt_dir}/phase_b_best.pt")
        logger.info(f"2. Compare against V7.3 baseline and SOTA methods")


if __name__ == '__main__':
    main()
