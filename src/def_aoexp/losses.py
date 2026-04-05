"""Loss functions for AO-Exp adversarial attack."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F  # noqa: N812
from torchvision.ops import box_iou

logger = logging.getLogger(__name__)

# Try to load CUDA fused mask CE kernel
_cuda_fused_mask_ce = None
try:
    from aoexp_cuda_kernels import fused_mask_ce as _cuda_fused_mask_ce
    logger.info("CUDA fused_mask_ce kernel loaded")
except ImportError:
    pass


def mask_cross_entropy_losses(
    adv_mask: torch.Tensor, clean_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute foreground and background CE losses between adversarial and clean masks.

    Uses custom CUDA kernel when available for single-pass computation.

    Args:
        adv_mask: Adversarial prediction mask, shape (H, W) in [0, 1].
        clean_mask: Clean prediction mask, shape (H, W) in [0, 1].

    Returns:
        (loss_fg, loss_bg): Cross-entropy losses for foreground and background pixels.
    """
    if _cuda_fused_mask_ce is not None and adv_mask.is_cuda:
        # CUDA kernel: single-pass FG/BG CE with block reduction
        losses = _cuda_fused_mask_ce(
            adv_mask.contiguous().flatten(),
            clean_mask.contiguous().flatten(),
        )
        return losses[0].squeeze(), losses[1].squeeze()

    # PyTorch fallback
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
    if x.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got {x.dim()}D")
    # Detect format: (H, W, C) has last dim small, (C, H, W) has first dim small
    if x.shape[-1] <= x.shape[0] and x.shape[-1] <= x.shape[1]:
        # (H, W, C) format
        c = x.shape[-1]
        return torch.mean(
            torch.stack([torch.linalg.svdvals(x[:, :, ch]).sum() for ch in range(c)])
        )
    # (C, H, W) format
    return torch.mean(
        torch.stack([torch.linalg.svdvals(x[c]).sum() for c in range(x.shape[0])])
    )


def mean_absolute_perturbation(delta: torch.Tensor) -> float:
    """MAP metric: mean per-pixel absolute perturbation averaged over spatial dims.

    Matches reference: np.sum(np.mean(np.abs(Result), axis=2)) / (H * W)
    """
    if delta.dim() == 3 and delta.shape[-1] <= delta.shape[0]:
        # (H, W, C)
        h, w = delta.shape[0], delta.shape[1]
        return torch.mean(torch.abs(delta), dim=2).sum().item() / (h * w)
    # (C, H, W)
    h, w = delta.shape[1], delta.shape[2]
    return torch.mean(torch.abs(delta), dim=0).sum().item() / (h * w)
