"""Export pipeline: pth, safetensors, ONNX, TensorRT FP16/FP32."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def export_pth(perturbation: torch.Tensor, output_dir: Path, metadata: dict | None = None):
    """Save perturbation as .pth checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "perturbation.pth"
    payload = {"perturbation": perturbation.cpu(), "metadata": metadata or {}}
    torch.save(payload, path)
    logger.info("Saved .pth: %s", path)
    return path


def export_safetensors(perturbation: torch.Tensor, output_dir: Path):
    """Save perturbation as safetensors."""
    from safetensors.torch import save_file

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "perturbation.safetensors"
    save_file({"perturbation": perturbation.cpu().contiguous()}, str(path))
    logger.info("Saved safetensors: %s", path)
    return path


class PerturbationApplier(torch.nn.Module):
    """Wraps perturbation as a module for ONNX/TRT export.

    Takes image (C, H, W) and applies universal perturbation.
    """

    def __init__(self, perturbation: torch.Tensor):
        super().__init__()
        # perturbation shape: (H, W, C) -> register as (C, H, W)
        if perturbation.dim() == 3 and perturbation.shape[-1] <= 4:
            perturbation = perturbation.permute(2, 0, 1)
        self.register_buffer("delta", perturbation)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return (image + self.delta).clamp(0.0, 1.0)


def export_onnx(
    perturbation: torch.Tensor,
    output_dir: Path,
    opset: int = 17,
    image_size: tuple[int, int] = (640, 640),
):
    """Export perturbation applier as ONNX model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model = PerturbationApplier(perturbation.clone().cpu())
    model.eval()
    model.cpu()

    h, w = image_size
    dummy = torch.randn(3, h, w)
    path = output_dir / "perturbation_applier.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(path),
        opset_version=max(opset, 18),
        input_names=["image"],
        output_names=["adversarial_image"],
        dynamo=False,
    )
    logger.info("Saved ONNX: %s", path)
    return path


def export_tensorrt(
    onnx_path: Path,
    output_dir: Path,
    precision: str = "fp16",
    workspace_gb: int = 4,
    image_size: tuple[int, int] = (640, 640),
):
    """Export ONNX model to TensorRT engine.

    Args:
        onnx_path: Path to ONNX model.
        output_dir: Output directory.
        precision: 'fp16' or 'fp32'.
        workspace_gb: Max workspace in GB.
        image_size: (H, W) for optimization profile.
    """
    try:
        import tensorrt as trt
    except ImportError:
        # Fallback to trtexec CLI
        logger.warning("TensorRT Python not available, using trtexec CLI")
        return _export_trt_cli(onnx_path, output_dir, precision, workspace_gb)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error("TRT parse error: %s", parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX for TensorRT")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    h, w = image_size
    profile = builder.create_optimization_profile()
    profile.set_shape("image", (3, h, w), (3, h, w), (3, h, w))
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"perturbation_applier_{precision}.engine"
    with open(path, "wb") as f:
        f.write(engine_bytes)
    logger.info("Saved TensorRT %s engine: %s", precision, path)
    return path


def _export_trt_cli(
    onnx_path: Path, output_dir: Path, precision: str, workspace_gb: int
) -> Path:
    """Fallback TensorRT export using trtexec command-line tool."""
    import subprocess

    output_dir.mkdir(parents=True, exist_ok=True)
    engine_path = output_dir / f"perturbation_applier_{precision}.engine"

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace_gb * 1024}",
    ]
    if precision == "fp16":
        cmd.append("--fp16")

    logger.info("Running trtexec: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        # Try shared TRT export tool
        shared_trt = Path("/mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py")
        if shared_trt.exists():
            logger.info("Using shared TRT toolkit: %s", shared_trt)
            cmd2 = [
                "python",
                str(shared_trt),
                "--onnx",
                str(onnx_path),
                "--output",
                str(engine_path),
                "--precision",
                precision,
            ]
            result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=600)
            if result2.returncode != 0:
                logger.error("TRT export failed: %s", result2.stderr)
                raise RuntimeError(f"TensorRT export failed: {result2.stderr}")
        else:
            logger.error("trtexec failed: %s", result.stderr)
            raise RuntimeError(f"trtexec failed: {result.stderr}")

    logger.info("Saved TensorRT %s engine: %s", precision, engine_path)
    return engine_path


def run_full_export(
    perturbation: torch.Tensor,
    output_dir: str | Path,
    metadata: dict | None = None,
    onnx_opset: int = 17,
    trt_workspace_gb: int = 4,
    image_size: tuple[int, int] = (640, 640),
) -> dict[str, Path]:
    """Run complete export pipeline: pth + safetensors + ONNX + TRT FP16 + TRT FP32."""
    output_dir = Path(output_dir)
    results = {}

    logger.info("=== Starting full export pipeline ===")

    # 1. PyTorch checkpoint
    results["pth"] = export_pth(perturbation, output_dir, metadata)

    # 2. SafeTensors
    results["safetensors"] = export_safetensors(perturbation, output_dir)

    # 3. ONNX
    onnx_path = export_onnx(perturbation, output_dir, opset=onnx_opset, image_size=image_size)
    results["onnx"] = onnx_path

    # 4. TensorRT FP32
    try:
        results["trt_fp32"] = export_tensorrt(
            onnx_path, output_dir, precision="fp32", workspace_gb=trt_workspace_gb, image_size=image_size
        )
    except Exception as e:
        logger.error("TRT FP32 export failed: %s", e)
        results["trt_fp32"] = None

    # 5. TensorRT FP16
    try:
        results["trt_fp16"] = export_tensorrt(
            onnx_path, output_dir, precision="fp16", workspace_gb=trt_workspace_gb, image_size=image_size
        )
    except Exception as e:
        logger.error("TRT FP16 export failed: %s", e)
        results["trt_fp16"] = None

    logger.info("=== Export pipeline complete ===")
    for fmt, path in results.items():
        logger.info("  %s: %s", fmt, path)

    return results
