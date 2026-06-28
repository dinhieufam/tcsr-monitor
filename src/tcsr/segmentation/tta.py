"""Test-time augmentation variance."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


_TTA_TRANSFORMS = [
    lambda x: x,
    lambda x: TF.hflip(x),
    lambda x: TF.vflip(x),
    lambda x: TF.hflip(TF.vflip(x)),
    lambda x: torch.rot90(x, 1, [2, 3]),
    lambda x: torch.rot90(x, 2, [2, 3]),
    lambda x: torch.rot90(x, 3, [2, 3]),
    lambda x: TF.adjust_brightness(x, 0.8),
]

_TTA_INVERSE = [
    lambda x: x,
    lambda x: TF.hflip(x),
    lambda x: TF.vflip(x),
    lambda x: TF.hflip(TF.vflip(x)),
    lambda x: torch.rot90(x, 3, [2, 3]),
    lambda x: torch.rot90(x, 2, [2, 3]),
    lambda x: torch.rot90(x, 1, [2, 3]),
    lambda x: x,
]


class TTARunner:
    def __init__(self, n_augmentations: int = 8):
        self.n = min(n_augmentations, len(_TTA_TRANSFORMS))

    @torch.no_grad()
    def __call__(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Return pixel-wise variance of sigmoid(logit) across TTA passes."""
        preds = []
        for fwd, inv in zip(_TTA_TRANSFORMS[: self.n], _TTA_INVERSE[: self.n]):
            aug = fwd(x)
            logits = model(aug)
            if hasattr(logits, "logits"):
                logits = logits.logits
            prob = torch.sigmoid(logits).squeeze(1)
            preds.append(inv(prob.unsqueeze(1)).squeeze(1))

        stack = torch.stack(preds, dim=0)  # (n, B, H, W)
        return stack.var(dim=0)            # (B, H, W)
