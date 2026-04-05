"""Data pipeline for COCO images and pre-computed DINOv2 features."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class COCOImageDataset(Dataset):
    """Loads COCO images and optionally pairs them with pre-computed DINOv2 features.

    Args:
        coco_root: Path to COCO dataset root (contains val2017/, etc.).
        split: Image split directory name (e.g., 'val2017').
        dinov2_features_dir: Path to pre-computed DINOv2 feature cache (optional).
        max_images: Maximum number of images to load. 0 = all.
        image_size: Target (H, W) for resizing.
    """

    def __init__(
        self,
        coco_root: str | Path,
        split: str = "val2017",
        dinov2_features_dir: str | Path | None = None,
        max_images: int = 500,
        image_size: tuple[int, int] = (640, 640),
    ):
        self.coco_root = Path(coco_root)
        self.image_dir = self.coco_root / split
        self.image_size = image_size
        self.dinov2_dir = Path(dinov2_features_dir) if dinov2_features_dir else None

        # Collect image paths
        if not self.image_dir.exists():
            raise FileNotFoundError(f"COCO split directory not found: {self.image_dir}")

        all_images = sorted(self.image_dir.glob("*.jpg"))
        if max_images > 0:
            all_images = all_images[:max_images]
        self.image_paths = all_images

        logger.info("Loaded %d images from %s", len(self.image_paths), self.image_dir)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = TF.to_tensor(TF.resize(img, list(self.image_size)))

        result = {
            "image": img_tensor,
            "image_id": img_path.stem,
            "path": str(img_path),
        }

        # Load pre-computed DINOv2 features if available
        if self.dinov2_dir:
            feat_path = self.dinov2_dir / f"{img_path.stem}.pt"
            if feat_path.exists():
                result["dinov2_features"] = torch.load(feat_path, map_location="cpu", weights_only=True)

        return result


def load_coco_images(
    coco_root: str,
    split: str = "val2017",
    max_images: int = 500,
    image_size: tuple[int, int] = (640, 640),
) -> list[torch.Tensor]:
    """Load COCO images as a list of tensors for attack engine.

    Returns:
        List of (C, H, W) tensors in [0, 1].
    """
    dataset = COCOImageDataset(
        coco_root=coco_root,
        split=split,
        max_images=max_images,
        image_size=image_size,
    )
    return [dataset[i]["image"] for i in range(len(dataset))]


def load_dinov2_features(
    features_dir: str | Path, image_ids: list[str] | None = None
) -> dict[str, torch.Tensor]:
    """Load pre-computed DINOv2 features from shared cache.

    Args:
        features_dir: Directory containing .pt feature files.
        image_ids: Optional list of image IDs to load. None = load all.

    Returns:
        Dict mapping image_id -> feature tensor.
    """
    features_dir = Path(features_dir)
    features = {}

    if image_ids:
        for img_id in image_ids:
            feat_path = features_dir / f"{img_id}.pt"
            if feat_path.exists():
                features[img_id] = torch.load(feat_path, map_location="cpu", weights_only=True)
    else:
        for feat_path in sorted(features_dir.glob("*.pt")):
            features[feat_path.stem] = torch.load(feat_path, map_location="cpu", weights_only=True)

    logger.info("Loaded %d DINOv2 feature vectors", len(features))
    return features
