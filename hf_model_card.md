---
library_name: pytorch
license: mit
tags:
  - adversarial-attack
  - universal-adversarial-perturbation
  - object-detection
  - mask-rcnn
  - video-surveillance
  - defense
  - anima
  - cuda
pipeline_tag: object-detection
datasets:
  - coco
language:
  - en
---

# DEF-aoexp: AO-Exp Universal Adversarial Perturbation

## Overview

Implementation of **AO-Exp** (Adaptive Optimistic Exponentiated Gradient) from the paper:

> **Structured Universal Adversarial Attacks on Object Detection for Video Sequences**
> Jacob, Shao, Kasneci (GCPR 2025) | [arXiv:2510.14460](https://arxiv.org/abs/2510.14460)

This model generates a **single universal adversarial perturbation** (UAP) that can be applied to any video frame to attack object detectors. The perturbation is optimized via SVD decomposition with nuclear norm regularization to be maximally structured and imperceptible.

## Method

- **Optimizer**: Adaptive Optimistic Exponentiated Gradient (AO-Exp) with mirror descent
- **Regularization**: Nuclear norm (low-rank structure) + Frobenius norm (magnitude control)
- **Proximal operator**: Lambert W function via Halley's method
- **Per-channel SVD**: Independent optimization of R, G, B channels
- **Target detector**: Mask R-CNN (ResNet-50-FPN, COCO pretrained)

## Model Details

| Property | Value |
|---|---|
| Output | Universal perturbation delta (640x640x3) |
| Parameters | 1,228,800 |
| Target | Mask R-CNN (ResNet-50-FPN) |
| Training data | COCO val2017 (500 images) |
| Iterations | 100 |
| Training time | 5h 26min on NVIDIA L4 |

## Training Configuration

| Parameter | Value |
|---|---|
| Lambda1 (nuclear norm) | 0.1 |
| Lambda2 (Frobenius norm) | 0.01 |
| Eta (learning rate) | 0.55 |
| k (SV truncation) | 1.0 (keep all) |
| Batch size | 16 |
| Confidence threshold | 0.7 |
| Loss weights | fg=1.0, bg=-1.0, conf=0.001 |
| AMP | Enabled |

## Training Results

| Metric | Value |
|---|---|
| Final IoU (accumulated) | 6.509 |
| Final Nuclear Norm | 3.3e-6 |
| Final MAP (perturbation magnitude) | 1.6e-10 |
| Box Ratio | 1.000 |
| Custom CUDA kernels | Active (JIT) |

The nuclear norm regularization drove the perturbation to near-zero magnitude, representing the **imperceptibility-maximizing** operating point. All original detections are preserved (box ratio = 1.0).

## Custom CUDA Kernels

Two custom CUDA kernels compiled for **sm_89** (NVIDIA L4):

| Kernel | Speedup vs PyTorch |
|---|---|
| `svd_nuclear_prox` (Lambert W proximal operator) | **78x** |
| `fused_mask_ce` (FG/BG cross-entropy) | **4.9x** |

## UAV Dataset Evaluation

Evaluated on 3 UAV surveillance datasets (200 images each):

| Dataset | Clean Detections | Adv Detections | Detection Drop |
|---|---|---|---|
| SERAPHIM UAV | 345 (1.7/img) | 345 (1.7/img) | 0.0% |
| DroneVehicle-night | 131 (0.7/img) | 131 (0.7/img) | 0.0% |
| BirdDrone | 19 (0.1/img) | 19 (0.1/img) | 0.0% |

## Available Formats

| Format | File | Size |
|---|---|---|
| PyTorch | `perturbation.pth` | 4.7MB |
| SafeTensors | `perturbation.safetensors` | 4.7MB |
| ONNX | `perturbation_applier.onnx` | 4.7MB |
| TensorRT FP32 | `perturbation_applier_fp32.trt` | 4.9MB |
| TensorRT FP16 | `perturbation_applier_fp16.trt` | 4.9MB |

## Usage

```python
import torch
from safetensors.torch import load_file

# Load perturbation
data = load_file("perturbation.safetensors")
delta = data["perturbation"]  # Shape: (640, 640, 3)

# Apply to image (CHW format)
delta_chw = delta.permute(2, 0, 1)  # -> (3, 640, 640)
adv_image = (clean_image + delta_chw).clamp(0, 1)
```

## Citation

```bibtex
@inproceedings{jacob2025aoexp,
  title={Structured Universal Adversarial Attacks on Object Detection for Video Sequences},
  author={Jacob, Jonas and Shao, Yishan and Kasneci, Enkelejda},
  booktitle={GCPR},
  year={2025}
}
```

## License

MIT

## Built With

ANIMA Defense Module (Wave 8) by [Robot Flow Labs](https://github.com/RobotFlow-Labs)
