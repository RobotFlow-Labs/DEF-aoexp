"""Tests for data pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from def_aoexp.data_pipeline import COCOImageDataset


def test_coco_dataset_missing_dir():
    with pytest.raises(FileNotFoundError):
        COCOImageDataset(coco_root="/nonexistent/path")


def test_coco_dataset_exists():
    coco_root = Path("/mnt/forge-data/datasets/coco")
    if not (coco_root / "val2017").exists():
        pytest.skip("COCO val2017 not available")
    ds = COCOImageDataset(coco_root=str(coco_root), max_images=5)
    assert len(ds) == 5
    item = ds[0]
    assert "image" in item
    assert item["image"].shape[0] == 3  # C, H, W
