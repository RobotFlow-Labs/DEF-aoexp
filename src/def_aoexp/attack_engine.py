"""AO-Exp Attack Engine: orchestrates perturbation optimization against Mask R-CNN."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torchvision
from tqdm import tqdm

from .ao_exp_optimizer import AOExpOptimizer
from .config import AOExpConfig
from .losses import (
    compute_iou_metric,
    confidence_loss,
    mask_cross_entropy_losses,
    mean_absolute_perturbation,
)

logger = logging.getLogger(__name__)


class AOExpAttackEngine:
    """Orchestrates the AO-Exp universal adversarial perturbation attack.

    Args:
        config: Full AOExpConfig.
        images: List of image tensors, each (C, H, W) in [0, 1].
    """

    def __init__(self, config: AOExpConfig, images: list[torch.Tensor]):
        self.cfg = config
        self.device = torch.device(config.training.device)
        self.images = images

        # Load target detector
        self.model = getattr(torchvision.models.detection, config.model.detector)(
            weights="DEFAULT" if config.model.pretrained else None
        )
        self.model.eval()
        self.model.to(self.device)

        # Get image dimensions from first image
        c, h, w = images[0].shape
        self.image_shape = (h, w, c)

        # Initialize optimizer
        self.optimizer = AOExpOptimizer(
            shape=self.image_shape,
            lambda1=config.attack.lambda1,
            lambda2=config.attack.lambda2,
            eta=config.attack.eta,
            k=config.attack.k,
            lower=config.attack.delta_lower,
            upper=config.attack.delta_upper,
            device=config.training.device,
        )

        # Metrics tracking
        self.iou_history: list[float] = []
        self.nuc_norm_history: list[float] = []
        self.box_ratio_history: list[float] = []

    def _select_detections(self, pred: dict, threshold: float) -> torch.Tensor:
        """Select detections above threshold, optionally filtering by class.

        target_class=0 means attack ALL classes (universal attack on COCO).
        target_class>0 means attack only that specific class.
        """
        score_mask = pred["scores"] >= threshold
        target_cls = self.cfg.model.target_class
        if target_cls > 0:
            return score_mask & (pred["labels"] == target_cls)
        return score_mask

    def _compute_gradient(self) -> torch.Tensor:
        """Compute averaged gradient of attack loss across all image batches."""
        threshold = self.cfg.attack.confidence_threshold
        batch_size = self.cfg.attack.batch_size

        # Get current perturbation as (C, H, W)
        delta_chw = self.optimizer.get_perturbation()
        delta_chw = delta_chw.detach().requires_grad_(True)

        total_grad = torch.zeros_like(delta_chw)
        total_iou = 0.0
        n_gt_boxes = 0
        n_adv_boxes = 0
        num_batches = 0

        for batch_start in range(0, len(self.images), batch_size):
            batch_end = min(batch_start + batch_size, len(self.images))
            batch_imgs = self.images[batch_start:batch_end]

            if delta_chw.grad is not None:
                delta_chw.grad = None
            batch_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
            has_grad = False

            for img in batch_imgs:
                img = img.to(self.device)
                adv_img = (img + delta_chw).clamp(0, 1)

                # Clean predictions
                with torch.no_grad():
                    clean_pred = self.model([img])[0]

                clean_mask_sel = self._select_detections(clean_pred, threshold)
                clean_boxes = clean_pred["boxes"][clean_mask_sel].detach().cpu()
                n_gt_boxes += clean_mask_sel.sum().item()

                masks = clean_pred["masks"][clean_mask_sel]
                if masks.shape[0] == 0:
                    continue

                combined_mask = masks.sum(dim=0).squeeze(0).clamp(0, 1)

                # Adversarial predictions (with gradient)
                adv_pred = self.model([adv_img])[0]
                adv_mask_sel = self._select_detections(adv_pred, threshold)
                n_adv_boxes += adv_mask_sel.sum().item()

                if (
                    adv_pred["scores"].shape[0] == 0
                    or adv_pred["masks"][adv_mask_sel].shape[0] == 0
                ):
                    total_iou += 0.0
                    continue

                adv_combined = (
                    adv_pred["masks"][adv_mask_sel].sum(dim=0).clamp(0, 1)[0, :, :]
                )
                adv_boxes = adv_pred["boxes"][adv_mask_sel].detach().cpu()

                total_iou += compute_iou_metric(clean_boxes, adv_boxes).item()

                loss_fg, loss_bg = mask_cross_entropy_losses(adv_combined, combined_mask)
                conf_l = confidence_loss(adv_pred["scores"][adv_mask_sel])

                frame_loss = (
                    self.cfg.attack.loss_weight_bg * loss_bg
                    + self.cfg.attack.loss_weight_fg * loss_fg
                    + self.cfg.attack.loss_weight_conf * conf_l
                )
                if isinstance(frame_loss, torch.Tensor):
                    batch_loss = batch_loss + frame_loss
                    has_grad = True

            if has_grad:
                batch_loss.backward()
                total_grad += delta_chw.grad.detach()
                delta_chw.grad = None
            num_batches += 1
            # Free intermediate computation graphs to manage VRAM
            del batch_loss
            torch.cuda.empty_cache()

        if num_batches > 0:
            total_grad /= num_batches

        # Track metrics
        n_images = len(self.images)
        self.iou_history.append(total_iou / max(n_images, 1))
        self.nuc_norm_history.append(self.optimizer.nuclear_norm())
        ratio = n_adv_boxes / max(n_gt_boxes, 1)
        self.box_ratio_history.append(ratio)

        # Return gradient in (H, W, C) format for optimizer
        return total_grad.permute(1, 2, 0)

    def run(
        self,
        num_iterations: int | None = None,
        save_dir: str | Path | None = None,
        callback=None,
    ) -> torch.Tensor:
        """Run the full AO-Exp attack optimization loop.

        Args:
            num_iterations: Override iteration count from config.
            save_dir: Directory to save checkpoints.
            callback: Optional callable(iteration, metrics_dict).

        Returns:
            Final perturbation tensor, shape (H, W, C).
        """
        nit = num_iterations or self.cfg.attack.num_iterations
        save_interval = self.cfg.training.save_interval

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting AO-Exp attack: %d iterations", nit)

        for it in tqdm(range(1, nit + 1), desc="AO-Exp Attack"):
            grad = self._compute_gradient()
            delta = self.optimizer.step(grad)

            metrics = {
                "iteration": it,
                "iou": self.iou_history[-1],
                "nuclear_norm": self.nuc_norm_history[-1],
                "box_ratio": self.box_ratio_history[-1],
                "map": mean_absolute_perturbation(delta),
            }

            if it % 10 == 0 or it == nit:
                logger.info(
                    "Iter %d/%d | IoU=%.4f | NucNorm=%.2f | BoxRatio=%.4f",
                    it,
                    nit,
                    metrics["iou"],
                    metrics["nuclear_norm"],
                    metrics["box_ratio"],
                )

            if save_dir and (it % save_interval == 0 or it == nit):
                ckpt = {
                    "iteration": it,
                    "perturbation": delta.cpu(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "metrics": metrics,
                    "iou_history": self.iou_history,
                    "nuc_norm_history": self.nuc_norm_history,
                    "box_ratio_history": self.box_ratio_history,
                }
                torch.save(ckpt, save_dir / f"checkpoint_iter{it:04d}.pth")

            if callback:
                callback(it, metrics)

        return delta

    def get_results(self) -> dict:
        """Return final attack results and metrics."""
        delta = self.optimizer.y
        return {
            "perturbation": delta.cpu(),
            "perturbation_chw": self.optimizer.get_perturbation().cpu(),
            "iou_history": self.iou_history,
            "nuc_norm_history": self.nuc_norm_history,
            "box_ratio_history": self.box_ratio_history,
            "final_iou": self.iou_history[-1] if self.iou_history else 0.0,
            "final_nuclear_norm": self.nuc_norm_history[-1] if self.nuc_norm_history else 0.0,
            "final_box_ratio": self.box_ratio_history[-1] if self.box_ratio_history else 1.0,
            "final_map": mean_absolute_perturbation(delta),
        }
