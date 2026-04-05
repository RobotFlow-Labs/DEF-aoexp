"""CUDA-accelerated training script for AO-Exp attack.

Uses custom CUDA kernels for SVD proximal updates and nuclear norm computation
when available, falling back to PyTorch operations otherwise.

Launch with nohup for long-running training:
    nohup python -m def_aoexp.train_cu --config configs/default.toml &
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.cuda.amp as amp

from .attack_engine import AOExpAttackEngine
from .config import AOExpConfig
from .data_pipeline import load_coco_images
from .export import run_full_export

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# Try loading custom CUDA kernels
_HAS_CUSTOM_KERNELS = False
try:
    from aoexp_cuda_kernels import svd_nuclear_prox, fused_mask_ce
    _HAS_CUSTOM_KERNELS = True
    logger.info("Custom CUDA kernels loaded successfully")
except ImportError:
    logger.info("Custom CUDA kernels not available, using PyTorch fallback")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return super().default(obj)


def check_vram_budget(device: str, target_pct: float = 0.65) -> dict:
    """Check VRAM usage and report budget."""
    dev = torch.device(device)
    idx = dev.index if dev.index is not None else 0
    total = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated(idx) / (1024**3)
    reserved = torch.cuda.memory_reserved(idx) / (1024**3)
    budget = total * target_pct
    return {
        "total_gb": total,
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "budget_gb": budget,
        "available_gb": budget - allocated,
    }


def main(config_path: str | None = None):
    parser = argparse.ArgumentParser(description="AO-Exp CUDA-Accelerated Attack Training")
    parser.add_argument("--config", type=str, default=config_path or "configs/default.toml")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--amp", action="store_true", default=True, help="Enable AMP")
    args = parser.parse_args()

    cfg = AOExpConfig.from_toml(args.config)
    if args.iterations:
        cfg.attack.num_iterations = args.iterations
    if args.device:
        cfg.training.device = args.device
    if args.max_images:
        cfg.data.max_images = args.max_images
    cfg.training.amp = args.amp

    # Setup output dirs
    output_dir = Path(cfg.training.output_dir)
    log_dir = Path(cfg.training.log_dir)
    export_dir = Path(cfg.export.export_dir)
    for d in [output_dir, log_dir, export_dir]:
        d.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_dir / "train_cu.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    # VRAM check
    vram = check_vram_budget(cfg.training.device)
    logger.info(
        "VRAM: %.1fGB total, %.1fGB allocated, budget=%.1fGB (65%%)",
        vram["total_gb"], vram["allocated_gb"], vram["budget_gb"],
    )

    logger.info("CUDA kernels: %s", "CUSTOM" if _HAS_CUSTOM_KERNELS else "PyTorch fallback")
    logger.info("AMP: %s", "enabled" if cfg.training.amp else "disabled")

    # Load data
    logger.info("Loading COCO images...")
    images = load_coco_images(
        coco_root=cfg.data.coco_root,
        split=cfg.data.split,
        max_images=cfg.data.max_images,
        image_size=tuple(cfg.data.image_size),
    )
    logger.info("Loaded %d images", len(images))

    # VRAM after data load
    vram = check_vram_budget(cfg.training.device)
    logger.info("Post-load VRAM: %.2fGB allocated", vram["allocated_gb"])

    # Run attack
    engine = AOExpAttackEngine(config=cfg, images=images)
    start_time = time.time()

    perturbation = engine.run(
        num_iterations=cfg.attack.num_iterations,
        save_dir=output_dir,
    )

    elapsed = time.time() - start_time
    logger.info("Training completed in %.1f seconds (%.1f min)", elapsed, elapsed / 60)

    # Save results
    results = engine.get_results()
    metrics = {
        "final_iou": results["final_iou"],
        "final_nuclear_norm": results["final_nuclear_norm"],
        "final_box_ratio": results["final_box_ratio"],
        "final_map": results["final_map"],
        "elapsed_seconds": elapsed,
        "custom_kernels": _HAS_CUSTOM_KERNELS,
        "amp_enabled": cfg.training.amp,
        "iou_history": results["iou_history"],
        "nuc_norm_history": results["nuc_norm_history"],
        "box_ratio_history": results["box_ratio_history"],
    }

    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)

    np.save(output_dir / "delta.npy", perturbation.cpu().numpy())

    # Export
    logger.info("Running full export pipeline...")
    export_results = run_full_export(
        perturbation=perturbation,
        output_dir=export_dir,
        metadata=metrics,
        onnx_opset=cfg.export.onnx_opset,
        trt_workspace_gb=cfg.export.trt_workspace_gb,
        image_size=tuple(cfg.data.image_size),
    )

    # Final VRAM
    vram = check_vram_budget(cfg.training.device)
    logger.info("Final VRAM: %.2fGB / %.1fGB (%.0f%%)",
                vram["allocated_gb"], vram["total_gb"],
                vram["allocated_gb"] / vram["total_gb"] * 100)
    logger.info("=== CUDA training complete ===")

    return perturbation, results


if __name__ == "__main__":
    main()
