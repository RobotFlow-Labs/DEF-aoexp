"""Tests for export pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from def_aoexp.export import PerturbationApplier, export_onnx, export_pth, export_safetensors


@pytest.fixture
def dummy_perturbation():
    return torch.randn(640, 640, 3) * 0.01


@pytest.fixture
def tmp_export_dir(tmp_path):
    return tmp_path / "exports"


def test_perturbation_applier(dummy_perturbation):
    model = PerturbationApplier(dummy_perturbation)
    img = torch.rand(3, 640, 640)
    out = model(img)
    assert out.shape == img.shape
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_export_pth(dummy_perturbation, tmp_export_dir):
    path = export_pth(dummy_perturbation, tmp_export_dir)
    assert path.exists()
    loaded = torch.load(path, weights_only=False)
    assert "perturbation" in loaded


def test_export_safetensors(dummy_perturbation, tmp_export_dir):
    path = export_safetensors(dummy_perturbation, tmp_export_dir)
    assert path.exists()


def test_export_onnx(dummy_perturbation, tmp_export_dir):
    path = export_onnx(dummy_perturbation, tmp_export_dir, image_size=(640, 640))
    assert path.exists()
    assert path.suffix == ".onnx"
