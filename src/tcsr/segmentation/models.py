"""Segmentation model factory (smp + transformers)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def build_model(cfg: Any) -> nn.Module:
    """Build a segmentation model from a Hydra seg_model config node."""
    arch = cfg.architecture

    if arch in ("Unet", "UnetPlusPlus", "DeepLabV3Plus", "FPN", "PAN", "PSPNet", "Linknet"):
        return _build_smp(cfg)
    elif arch == "segformer":
        return _build_segformer(cfg)
    elif arch == "sam2":
        return _build_sam2(cfg)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def _build_smp(cfg: Any) -> nn.Module:
    import segmentation_models_pytorch as smp

    model_cls = getattr(smp, cfg.architecture)
    model = model_cls(
        encoder_name=cfg.encoder,
        encoder_weights=cfg.encoder_weights,
        in_channels=cfg.in_channels,
        classes=cfg.classes,
        activation=None,
    )
    if cfg.checkpoint:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state)
    return model


def _build_segformer(cfg: Any) -> nn.Module:
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(
        cfg.hf_model,
        num_labels=cfg.num_labels,
        ignore_mismatched_sizes=True,
    )
    if cfg.checkpoint:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state)
    return model


def _build_sam2(cfg: Any) -> nn.Module:
    from transformers import SamModel

    model = SamModel.from_pretrained(cfg.model_id)
    return model


def get_encoder(model: nn.Module) -> nn.Module:
    """Return the backbone/encoder for embedding extraction."""
    if hasattr(model, "encoder"):
        return model.encoder
    raise AttributeError("Model has no .encoder attribute")
