"""Light training loop for the base segmentation model (optional stage)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from tcsr.utils.logging import get_logger

log = get_logger(__name__)


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(pred).flatten(1)
    target = target.float().flatten(1)
    intersection = (pred * target).sum(1)
    return 1 - (2 * intersection + eps) / (pred.sum(1) + target.sum(1) + eps)


def bce_dice_loss(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits.squeeze(1), masks.float())
    return bce + dice_loss(logits.squeeze(1), masks).mean()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        if hasattr(logits, "logits"):
            logits = logits.logits
        loss = bce_dice_loss(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    dice_sum = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        if hasattr(logits, "logits"):
            logits = logits.logits
        pred = (torch.sigmoid(logits.squeeze(1)) >= 0.5).float()
        intersection = (pred * masks).sum((1, 2))
        dice = (2 * intersection + 1e-6) / (pred.sum((1, 2)) + masks.sum((1, 2)) + 1e-6)
        dice_sum += dice.mean().item()
    return dice_sum / len(loader)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint_path: Path,
    max_epochs: int = 30,
    lr: float = 1e-4,
    patience: int = 5,
    device: str = "cuda",
) -> dict:
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_dice = 0.0
    no_improve = 0
    metrics = []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_dice = validate(model, val_loader, device)
        scheduler.step()
        metrics.append({"epoch": epoch, "train_loss": train_loss, "val_dice": val_dice})
        log.info("Epoch %d | loss=%.4f | val_dice=%.4f", epoch, train_loss, val_dice)

        if val_dice > best_dice:
            best_dice = val_dice
            no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            log.info("  → saved checkpoint (dice=%.4f)", best_dice)
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("Early stopping at epoch %d", epoch)
                break

    return {"best_val_dice": best_dice, "epochs_trained": epoch, "history": metrics}
