# DEF-aoexp

ANIMA Defense Module — Structured Universal Adversarial Attacks on Object Detection for Video Sequences (AO-Exp).

Paper: [arXiv 2510.14460](https://arxiv.org/abs/2510.14460) | GCPR 2025

## Method

AO-Exp generates universal adversarial perturbations (UAP) using adaptive optimistic exponentiated gradient descent with nuclear norm regularization. A single perturbation is optimized to fool Mask R-CNN across video frames by concentrating structured, low-rank noise in the background.

## Usage

```bash
source /mnt/forge-data/activate.sh
CUDA_VISIBLE_DEVICES=0 uv run python -m def_aoexp.train_cu --config configs/default.toml
```

## Structure

```
src/def_aoexp/
  ├── ao_exp_optimizer.py  # Core AO-Exp optimizer (GPU PyTorch)
  ├── attack_engine.py     # Attack orchestration with Mask R-CNN
  ├── losses.py            # Loss functions (mask CE, confidence, IoU)
  ├── data_pipeline.py     # COCO + DINOv2 data loading
  ├── config.py            # Pydantic config from TOML
  ├── export.py            # Export: pth, safetensors, ONNX, TRT
  ├── train.py             # Standard training entry
  └── train_cu.py          # CUDA-accelerated training entry
kernels/
  └── aoexp_cuda_kernels.cu  # Custom CUDA: SVD proximal + fused mask CE
```
