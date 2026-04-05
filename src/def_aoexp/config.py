"""Configuration management using Pydantic BaseSettings + TOML."""

from __future__ import annotations

from pathlib import Path

import toml
from pydantic import Field
from pydantic_settings import BaseSettings


class AttackConfig(BaseSettings):
    num_iterations: int = 100
    confidence_threshold: float = 0.8
    lambda1: float = 0.1
    lambda2: float = 0.01
    eta: float = 0.55
    k: float | str = 1.0
    delta_lower: float = -1.0
    delta_upper: float = 1.0
    batch_size: int = 30
    loss_weight_fg: float = 1.0
    loss_weight_bg: float = -1.0
    loss_weight_conf: float = 0.001


class DataConfig(BaseSettings):
    coco_root: str = "/mnt/forge-data/datasets/coco"
    dinov2_features: str = "/mnt/forge-data/shared_infra/datasets/coco_dinov2_features"
    split: str = "val2017"
    max_images: int = 500
    image_size: list[int] = Field(default_factory=lambda: [640, 640])


class ModelConfig(BaseSettings):
    detector: str = "maskrcnn_resnet50_fpn"
    pretrained: bool = True
    target_class: int = 1


class TrainingConfig(BaseSettings):
    device: str = "cuda:0"
    amp: bool = True
    grad_accum_steps: int = 1
    save_interval: int = 25
    output_dir: str = "/mnt/artifacts-datai/checkpoints/DEF-aoexp"
    log_dir: str = "/mnt/artifacts-datai/logs/DEF-aoexp"


class ExportConfig(BaseSettings):
    export_dir: str = "/mnt/artifacts-datai/exports/DEF-aoexp"
    onnx_opset: int = 17
    trt_workspace_gb: int = 4


class AOExpConfig(BaseSettings):
    attack: AttackConfig = Field(default_factory=AttackConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> AOExpConfig:
        raw = toml.load(str(path))
        return cls(
            attack=AttackConfig(**raw.get("attack", {})),
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            export=ExportConfig(**raw.get("export", {})),
        )
