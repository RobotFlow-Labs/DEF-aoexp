"""Loss functions for AO-Exp adversarial attack."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou


def mask_cross_entropy_losses(
    adv_mask: torch.Tensor, clean_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute foreground and background CE losses between adversarial and clean masks.

    Args:
        adv_mask: Adversarial prediction mask, shape (H, W) in [0, 1].
        clean_mask: Clean prediction mask, shape (H, W) in [0, 1].

    Returns:
        (loss_fg, loss_bg): Cross-entropy losses for foreground and background pixels.
    """
    # Stack into 2-class probabilities: (H*W, 2)
    adv_probs = torch.stack([1.0 - adv_mask, adv_mask], dim=-1).reshape(-1, 2)
    target = ((clean_mask > 0).float()).flatten().long()

    fg_idx = target.nonzero(as_tuple=True)[0]
    bg_idx = (target == 0).nonzero(as_tuple=True)[0]

    loss_fg = F.cross_entropy(adv_probs[fg_idx], target[fg_idx]) if fg_idx.numel() > 0 else 0.0
    loss_bg = F.cross_entropy(adv_probs[bg_idx], target[bg_idx]) if bg_idx.numel() > 0 else 0.0

    return loss_fg, loss_bg


def confidence_loss(scores: torch.Tensor) -> torch.Tensor:
    """Penalty for high-confidence adversarial detections."""
    return torch.sum(scores * 100.0)


def compute_iou_metric(
    gt_boxes: torch.Tensor, adv_boxes: torch.Tensor
) -> torch.Tensor:
    """Compute accumulated IoU between ground-truth and adversarial boxes."""
    if adv_boxes.shape[0] == 0:
        return torch.tensor(0.0)
    iou_matrix = box_iou(gt_boxes, adv_boxes)
    return iou_matrix.sum()


def nuclear_norm(x: torch.Tensor) -> torch.Tensor:
    """Mean nuclear norm across channels. Input shape (C, H, W) or (H, W, C)."""
    if x.dim() == 3 and x.shape[0] <= 4:
        # (C, H, W) format
        return torch.mean(torch.stack([torch.linalg.svdvals(x[c]).sum() for c in range(x.shape[0])]))
    # (H, W, C) format
    c = x.shape[-1]
    return torch.mean(
        torch.stack([torch.linalg.svdvals(x[:, :, ch]).sum() for ch in range(c)])
    )


def mean_absolute_perturbation(delta: torch.Tensor) -> float:
    """MAP metric: mean absolute perturbation normalized by spatial dimensions."""
    if delta.dim() == 3 and delta.shape[-1] == 3:
        # (H, W, C)
        h, w = delta.shape[0], delta.shape[1]
    else:
        # (C, H, W)
        h, w = delta.shape[1], delta.shape[2]
    return (torch.mean(torch.abs(delta)).item() * delta.shape[-1]) / 1.0
