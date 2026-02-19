#!/usr/bin/env python3
"""
STF-Mamba V8.0 - Training Loop
=================================

Complete training pipeline:
    1. Train epoch with gradient clipping
    2. Validate with AUC computation
    3. Checkpoint best model by val AUC
    4. Log epoch table: Ep | TrLoss | TrAcc | VaLoss | VaAUC | VaAcc | VarGap | LR | Time

Handles:
    - DataParallel (multi-GPU): saves model.module.state_dict()
    - Mixed precision: NOT used (DINOv2 + Mamba stability)
    - Kaggle: num_workers=0, checkpoint to /kaggle/working/
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from stf_mamba.model import STFMambaV8
from stf_mamba.losses import STFMambaLoss
from training.optimizer import build_optimizer, clip_gradients
from training.scheduler import build_scheduler

logger = logging.getLogger(__name__)


class Trainer:
    """
    STF-Mamba V8.0 training loop.

    Args:
        model: STFMambaV8 model.
        criterion: STFMambaLoss instance.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Dict with training hyperparameters.
        save_dir: Directory for checkpoints.
        device: 'cuda' or 'cpu'.
    """

    def __init__(
        self,
        model: STFMambaV8,
        criterion: STFMambaLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        save_dir: str = "checkpoints",
        device: str = "cuda",
    ) -> None:
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)

        # Model — DataParallel if multi-GPU
        self.model = model.to(self.device)
        if torch.cuda.device_count() > 1:
            logger.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        self.is_parallel = isinstance(self.model, nn.DataParallel)

        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Build optimizer and scheduler
        base_model = self.model.module if self.is_parallel else self.model
        self.optimizer = build_optimizer(
            base_model,
            lr_backbone=config.get("lr_backbone", 5e-6),
            lr_temporal=config.get("lr_temporal", 1e-4),
            lr_head=config.get("lr_head", 1e-4),
            weight_decay=config.get("weight_decay", 1e-4),
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            total_epochs=config.get("epochs", 25),
            warmup_epochs=config.get("warmup_epochs", 3),
        )

        self.grad_clip = config.get("grad_clip", 1.0)
        self.best_val_auc = 0.0
        self.epoch = 0

        # Print header
        self._print_header()

    def train(self, num_epochs: Optional[int] = None) -> Dict:
        """
        Run full training loop.

        Args:
            num_epochs: Override config epochs if provided.

        Returns:
            Dict with training history.
        """
        num_epochs = num_epochs or self.config.get("epochs", 25)
        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_auc": [], "val_acc": [],
            "var_gap": [], "lr": [],
        }

        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            t0 = time.time()

            # Train
            train_metrics = self._train_epoch()

            # Validate
            val_metrics = self._validate_epoch()

            # Step scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[-1]["lr"]

            elapsed = time.time() - t0

            # Log
            self._print_epoch(
                train_metrics, val_metrics, current_lr, elapsed
            )

            # Checkpoint
            if val_metrics["auc"] > self.best_val_auc:
                self.best_val_auc = val_metrics["auc"]
                self._save_checkpoint("best.pth", val_metrics)

            if self.epoch % 10 == 0:
                self._save_checkpoint(f"epoch_{self.epoch}.pth", val_metrics)

            # History
            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["acc"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_auc"].append(val_metrics["auc"])
            history["val_acc"].append(val_metrics["acc"])
            history["var_gap"].append(val_metrics["var_gap"])
            history["lr"].append(current_lr)

        logger.info(f"\nTraining complete. Best val AUC: {self.best_val_auc:.4f}")
        return history

    def _train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_var_gap = 0.0
        num_batches = 0

        for batch in self.train_loader:
            frames = batch["frames"].to(self.device)   # (B, T, 3, H, W)
            labels = batch["label"].to(self.device)     # (B,)

            # Forward
            self.optimizer.zero_grad()
            output = self.model(frames)
            loss_dict = self.criterion(
                output["logits"], labels, output["variance"]
            )

            # Backward
            loss_dict["total"].backward()
            grad_norm = clip_gradients(
                self.model.module if self.is_parallel else self.model,
                self.grad_clip,
            )
            self.optimizer.step()

            # Metrics
            preds = output["logits"].argmax(dim=1)
            correct = (preds == labels).sum().item()
            bs = labels.size(0)

            total_loss += loss_dict["ce"].item() * bs
            total_correct += correct
            total_samples += bs
            total_var_gap += loss_dict["var_gap"].item()
            num_batches += 1

        return {
            "loss": total_loss / max(total_samples, 1),
            "acc": total_correct / max(total_samples, 1),
            "var_gap": total_var_gap / max(num_batches, 1),
        }

    @torch.no_grad()
    def _validate_epoch(self) -> Dict[str, float]:
        """Run one validation epoch with AUC computation."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_var_gap = 0.0
        num_batches = 0

        all_probs = []
        all_labels = []

        for batch in self.val_loader:
            frames = batch["frames"].to(self.device)
            labels = batch["label"].to(self.device)

            output = self.model(frames)
            loss_dict = self.criterion(
                output["logits"], labels, output["variance"]
            )

            # Collect for AUC
            probs = torch.softmax(output["logits"], dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            preds = output["logits"].argmax(dim=1)
            correct = (preds == labels).sum().item()
            bs = labels.size(0)

            total_loss += loss_dict["ce"].item() * bs
            total_correct += correct
            total_samples += bs
            total_var_gap += loss_dict["var_gap"].item()
            num_batches += 1

        # Compute AUC
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5  # Single class in batch

        return {
            "loss": total_loss / max(total_samples, 1),
            "acc": total_correct / max(total_samples, 1),
            "auc": auc,
            "var_gap": total_var_gap / max(num_batches, 1),
        }

    def _save_checkpoint(self, filename: str, val_metrics: Dict) -> None:
        """Save model checkpoint."""
        base_model = self.model.module if self.is_parallel else self.model
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": base_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_auc": self.best_val_auc,
            "val_metrics": val_metrics,
            "config": self.config,
        }
        path = self.save_dir / filename
        torch.save(checkpoint, path)
        if "best" in filename:
            logger.info(f"  ★ Saved best model: AUC={val_metrics['auc']:.4f}")

    def load_checkpoint(self, path: str) -> None:
        """Load a checkpoint to resume training."""
        checkpoint = torch.load(path, map_location=self.device)

        base_model = self.model.module if self.is_parallel else self.model
        base_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_auc = checkpoint.get("best_val_auc", 0.0)
        self.epoch = checkpoint.get("epoch", 0)

        logger.info(
            f"Resumed from epoch {self.epoch}, "
            f"best AUC={self.best_val_auc:.4f}"
        )

    def _print_header(self) -> None:
        """Print training table header."""
        header = (
            f"{'Ep':>4} | {'TrLoss':>7} | {'TrAcc':>6} | "
            f"{'VaLoss':>7} | {'VaAUC':>6} | {'VaAcc':>6} | "
            f"{'VarGap':>7} | {'LR':>9} | {'Time':>5}"
        )
        sep = "-" * len(header)
        print(f"\n{sep}")
        print(header)
        print(sep)

    def _print_epoch(
        self,
        train: Dict,
        val: Dict,
        lr: float,
        elapsed: float,
    ) -> None:
        """Print one row of the epoch table."""
        line = (
            f"{self.epoch:4d} | "
            f"{train['loss']:7.4f} | {train['acc']:6.3f} | "
            f"{val['loss']:7.4f} | {val['auc']:6.4f} | {val['acc']:6.3f} | "
            f"{val['var_gap']:+7.4f} | "
            f"{lr:9.2e} | {elapsed:5.1f}s"
        )
        best_marker = " ★" if val["auc"] >= self.best_val_auc else ""
        print(line + best_marker)
