"""Training entry point for AO-Exp adversarial perturbation generation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from .attack_engine import AOExpAttackEngine
from .config import AOExpConfig
from .data_pipeline import load_coco_images
from .export import run_full_export
from .utils import NumpyEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main(config_path: str | None = None):
    parser = argparse.ArgumentParser(description="AO-Exp Attack Training")
    parser.add_argument(
        "--config",
        type=str,
        default=config_path or "configs/default.toml",
        help="Path to TOML config file",
    )
    parser.add_argument("--iterations", type=int, default=None, help="Override num_iterations")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--max-images", type=int, default=None, help="Override max_images")
    args = parser.parse_args()

    # Load config
    cfg = AOExpConfig.from_toml(args.config)
    if args.iterations:
        cfg.attack.num_iterations = args.iterations
    if args.device:
        cfg.training.device = args.device
    if args.max_images:
        cfg.data.max_images = args.max_images

    logger.info("Config loaded: lambda1=%.3f, lambda2=%.3f, eta=%.3f, k=%s",
                cfg.attack.lambda1, cfg.attack.lambda2, cfg.attack.eta, cfg.attack.k)

    # Create output directories
    output_dir = Path(cfg.training.output_dir)
    log_dir = Path(cfg.training.log_dir)
    export_dir = Path(cfg.export.export_dir)
    for d in [output_dir, log_dir, export_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Add file logger
    fh = logging.FileHandler(log_dir / "train.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    # Load data
    logger.info("Loading COCO images from %s/%s (max=%d)",
                cfg.data.coco_root, cfg.data.split, cfg.data.max_images)
    images = load_coco_images(
        coco_root=cfg.data.coco_root,
        split=cfg.data.split,
        max_images=cfg.data.max_images,
        image_size=tuple(cfg.data.image_size),
    )
    logger.info("Loaded %d images", len(images))

    # Run attack
    engine = AOExpAttackEngine(config=cfg, images=images)
    perturbation = engine.run(
        num_iterations=cfg.attack.num_iterations,
        save_dir=output_dir,
    )

    # Save results
    results = engine.get_results()
    metrics_path = output_dir / "final_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "final_iou": results["final_iou"],
                "final_nuclear_norm": results["final_nuclear_norm"],
                "final_box_ratio": results["final_box_ratio"],
                "final_map": results["final_map"],
                "iou_history": results["iou_history"],
                "nuc_norm_history": results["nuc_norm_history"],
                "box_ratio_history": results["box_ratio_history"],
            },
            f,
            indent=2,
            cls=NumpyEncoder,
        )
    logger.info("Metrics saved: %s", metrics_path)

    # Save raw perturbation
    np.save(output_dir / "delta.npy", perturbation.cpu().numpy())
    logger.info("Perturbation saved: %s", output_dir / "delta.npy")

    # Full export pipeline
    logger.info("Running export pipeline...")
    run_full_export(
        perturbation=perturbation,
        output_dir=export_dir,
        metadata=results,
        onnx_opset=cfg.export.onnx_opset,
        trt_workspace_gb=cfg.export.trt_workspace_gb,
        image_size=tuple(cfg.data.image_size),
    )

    logger.info("=== Training complete ===")
    logger.info("Final IoU: %.4f", results["final_iou"])
    logger.info("Final Nuclear Norm: %.2f", results["final_nuclear_norm"])
    logger.info("Final Box Ratio: %.4f", results["final_box_ratio"])

    return perturbation, results


if __name__ == "__main__":
    main()
