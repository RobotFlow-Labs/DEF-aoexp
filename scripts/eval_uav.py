"""Evaluate AO-Exp universal adversarial perturbation on UAV datasets.

Applies the trained perturbation to UAV imagery and measures attack effectiveness
by comparing clean vs adversarial Mask R-CNN detections.

Datasets:
  1. SERAPHIM UAV (8.3K test images) — aerial surveillance
  2. DroneVehicle-night (34K images) — night-time drone IR/RGB
  3. BirdDrone (145K images) — bird vs drone classification

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/eval_uav.py --dataset seraphim --max-images 500
    CUDA_VISIBLE_DEVICES=4 python scripts/eval_uav.py --dataset all --max-images 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


DATASET_CONFIGS = {
    "seraphim": {
        "name": "SERAPHIM UAV",
        "path": "/mnt/forge-data/datasets/uav_detection/seraphim/test/images",
        "ext": "*.jpg",
        "description": "Aerial surveillance UAV imagery",
    },
    "dronevehicle": {
        "name": "DroneVehicle-night RGB",
        "path": "/mnt/forge-data/shared_infra/datasets/dronevehicle_night/DroneVehicle-night/rgb",
        "ext": "**/*.jpg",
        "description": "Night-time drone RGB vehicle detection",
    },
    "birddrone": {
        "name": "BirdDrone",
        "path": "/mnt/forge-data/shared_infra/datasets/lat_birddrone/drone2025/dronepicture/images",
        "ext": "**/*.jpg",
        "description": "Bird vs drone aerial objects",
    },
}


def load_perturbation(checkpoint_dir: str, device: str) -> torch.Tensor:
    """Load trained perturbation from checkpoint."""
    ckpt_dir = Path(checkpoint_dir)

    # Try safetensors first, then pth
    st_path = ckpt_dir.parent / "exports" / "DEF-aoexp" / "perturbation.safetensors"
    pth_path = ckpt_dir.parent / "exports" / "DEF-aoexp" / "perturbation.pth"
    npy_path = ckpt_dir / "delta.npy"

    if st_path.exists():
        from safetensors.torch import load_file
        data = load_file(str(st_path))
        delta = data["perturbation"].to(device)
        logger.info("Loaded perturbation from safetensors: %s", delta.shape)
    elif pth_path.exists():
        data = torch.load(pth_path, map_location=device, weights_only=True)
        delta = data["perturbation"].to(device)
        logger.info("Loaded perturbation from pth: %s", delta.shape)
    elif npy_path.exists():
        arr = np.load(npy_path)
        delta = torch.from_numpy(arr).to(device)
        logger.info("Loaded perturbation from npy: %s", delta.shape)
    else:
        raise FileNotFoundError(f"No perturbation found in {ckpt_dir}")

    # Ensure (C, H, W) format
    if delta.dim() == 3 and delta.shape[-1] == 3:
        delta = delta.permute(2, 0, 1)

    return delta.float()


def load_images(
    dataset_key: str, max_images: int, image_size: tuple[int, int] = (640, 640)
) -> list[torch.Tensor]:
    """Load images from a UAV dataset."""
    cfg = DATASET_CONFIGS[dataset_key]
    root = Path(cfg["path"])

    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")

    pattern = cfg["ext"]
    img_paths = sorted(root.glob(pattern))
    if max_images > 0:
        img_paths = img_paths[:max_images]

    logger.info("Loading %d images from %s (%s)", len(img_paths), cfg["name"], root)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    images = []
    for p in tqdm(img_paths, desc=f"Loading {cfg['name']}", leave=False):
        try:
            img = Image.open(p).convert("RGB")
            images.append(transform(img))
        except Exception:
            continue

    logger.info("Loaded %d valid images", len(images))
    return images


def evaluate_attack(
    model: torch.nn.Module,
    images: list[torch.Tensor],
    delta: torch.Tensor,
    device: str,
    threshold: float = 0.5,
    batch_size: int = 8,
) -> dict:
    """Evaluate adversarial perturbation effectiveness.

    Metrics:
    - detection_drop_rate: fraction of detections removed by perturbation
    - avg_confidence_drop: average confidence reduction
    - attack_success_rate: fraction of images where detections decrease
    - avg_clean_detections: baseline detection count
    - avg_adv_detections: adversarial detection count
    """
    model.eval()
    delta_dev = delta.to(device)

    # Resize delta to match image size if needed
    if images and images[0].shape[1:] != delta_dev.shape[1:]:
        delta_dev = torch.nn.functional.interpolate(
            delta_dev.unsqueeze(0),
            size=images[0].shape[1:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    total_clean_dets = 0
    total_adv_dets = 0
    total_clean_conf = 0.0
    total_adv_conf = 0.0
    images_attacked = 0
    images_with_drop = 0

    for i in tqdm(range(0, len(images), batch_size), desc="Evaluating"):
        batch = images[i : i + batch_size]

        for img in batch:
            img_dev = img.to(device)

            # Clean predictions
            with torch.no_grad():
                clean_pred = model([img_dev])[0]
            clean_mask = clean_pred["scores"] >= threshold
            n_clean = clean_mask.sum().item()
            conf_clean = clean_pred["scores"][clean_mask].sum().item() if n_clean > 0 else 0.0

            # Adversarial predictions
            adv_img = (img_dev + delta_dev).clamp(0, 1)
            with torch.no_grad():
                adv_pred = model([adv_img])[0]
            adv_mask = adv_pred["scores"] >= threshold
            n_adv = adv_mask.sum().item()
            conf_adv = adv_pred["scores"][adv_mask].sum().item() if n_adv > 0 else 0.0

            total_clean_dets += n_clean
            total_adv_dets += n_adv
            total_clean_conf += conf_clean
            total_adv_conf += conf_adv
            images_attacked += 1

            if n_adv < n_clean:
                images_with_drop += 1

        torch.cuda.empty_cache()

    n = max(images_attacked, 1)
    avg_clean = total_clean_dets / n
    avg_adv = total_adv_dets / n
    det_drop = 1.0 - (total_adv_dets / max(total_clean_dets, 1))
    conf_drop = 1.0 - (total_adv_conf / max(total_clean_conf, 1e-6))
    asr = images_with_drop / n

    return {
        "images_evaluated": images_attacked,
        "total_clean_detections": total_clean_dets,
        "total_adv_detections": total_adv_dets,
        "avg_clean_detections": round(avg_clean, 2),
        "avg_adv_detections": round(avg_adv, 2),
        "detection_drop_rate": round(det_drop, 4),
        "confidence_drop_rate": round(conf_drop, 4),
        "attack_success_rate": round(asr, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="AO-Exp UAV Attack Evaluation")
    parser.add_argument(
        "--dataset",
        choices=["seraphim", "dronevehicle", "birddrone", "all"],
        default="all",
    )
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/mnt/artifacts-datai/checkpoints/DEF-aoexp",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/artifacts-datai/reports/DEF-aoexp",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load perturbation
    delta = load_perturbation(args.checkpoint_dir, args.device)
    logger.info(
        "Perturbation shape: %s, range: [%.4f, %.4f]",
        delta.shape, delta.min(), delta.max(),
    )

    # Load detector
    logger.info("Loading Mask R-CNN...")
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval()
    model.to(args.device)

    datasets = (
        list(DATASET_CONFIGS.keys()) if args.dataset == "all" else [args.dataset]
    )

    all_results = {}
    for ds_key in datasets:
        cfg = DATASET_CONFIGS[ds_key]
        logger.info("=" * 60)
        logger.info("Evaluating: %s", cfg["name"])
        logger.info("=" * 60)

        try:
            images = load_images(ds_key, args.max_images)
            if not images:
                logger.warning("No images loaded for %s, skipping", cfg["name"])
                continue

            start = time.time()
            results = evaluate_attack(
                model=model,
                images=images,
                delta=delta,
                device=args.device,
                threshold=args.threshold,
            )
            elapsed = time.time() - start
            results["eval_time_seconds"] = round(elapsed, 1)
            results["dataset"] = cfg["name"]

            all_results[ds_key] = results

            logger.info("Results for %s:", cfg["name"])
            logger.info("  Images: %d", results["images_evaluated"])
            clean_d = results["total_clean_detections"]
            adv_d = results["total_adv_detections"]
            logger.info(
                "  Clean dets: %d (avg %.1f/img)", clean_d, results["avg_clean_detections"],
            )
            logger.info(
                "  Adv dets: %d (avg %.1f/img)", adv_d, results["avg_adv_detections"],
            )
            logger.info("  Detection drop: %.1f%%", results["detection_drop_rate"] * 100)
            logger.info("  Confidence drop: %.1f%%", results["confidence_drop_rate"] * 100)
            logger.info("  Attack success rate: %.1f%%", results["attack_success_rate"] * 100)
            logger.info("  Eval time: %.1fs", elapsed)

            # Free memory
            del images
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error("Failed on %s: %s", cfg["name"], e)
            all_results[ds_key] = {"error": str(e)}

    # Save results
    out_path = output_dir / "uav_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("\nAll results saved to %s", out_path)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Dataset':<25} {'Images':>6} {'Det Drop':>10} {'Conf Drop':>10} {'ASR':>8}")
    print("-" * 70)
    for k, v in all_results.items():
        if "error" in v:
            print(f"{DATASET_CONFIGS[k]['name']:<25} {'ERROR':>6}")
        else:
            print(
                f"{v['dataset']:<25} {v['images_evaluated']:>6} "
                f"{v['detection_drop_rate']*100:>9.1f}% "
                f"{v['confidence_drop_rate']*100:>9.1f}% "
                f"{v['attack_success_rate']*100:>7.1f}%"
            )
    print("=" * 70)


if __name__ == "__main__":
    main()
