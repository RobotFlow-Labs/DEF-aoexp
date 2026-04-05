"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

from def_aoexp.config import AOExpConfig


def test_from_toml():
    config_path = Path(__file__).parent.parent / "configs" / "default.toml"
    cfg = AOExpConfig.from_toml(config_path)
    assert cfg.attack.num_iterations == 100
    assert cfg.attack.lambda1 == 0.1
    assert cfg.model.detector == "maskrcnn_resnet50_fpn"
    assert cfg.data.split == "val2017"


def test_defaults():
    cfg = AOExpConfig()
    assert cfg.attack.confidence_threshold == 0.8
    assert cfg.training.amp is True
    assert cfg.export.onnx_opset == 17
