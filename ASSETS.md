# DEF-aoexp — Asset Manifest

## Paper
- Title: Structured Universal Adversarial Attacks on Object Detection for Video Sequences
- ArXiv: 2510.14460
- Authors: Sven Jacob, Weijia Shao, Gjergji Kasneci
- Conference: GCPR 2025
- Code: https://github.com/jsve96/AO-Exp-Attack

## Status
- Module status: COMPLETE
- Gate 1 paper alignment: PASS
- Phase 4 build: COMPLETE
- Code review: PASS (ruff clean, 24/24 tests)
- Training: COMPLETE (50 iterations, 100 images)
- Export: COMPLETE (pth + safetensors + ONNX + TRT FP16 + TRT FP32)

## Source Verification
- Local reference repo: `repositories/AO-Exp-Attack/`
- Upstream datasets: PETS 2009 S2L1, EPFL-RLC, CW4C (video surveillance)
- Adapted to: COCO val2017

## Pretrained Weights
| Model | Source | Server Path | Size | Status |
|---|---|---|---|---|
| Mask R-CNN (ResNet-50-FPN) | torchvision | (downloaded on demand) | ~170MB | READY |
| DINOv2 ViT-B/14 | facebook | /mnt/forge-data/models/dinov2_vitb14_pretrain.pth | 346MB | READY |

## Datasets
| Dataset | Scope | Server Path | Size | Status |
|---|---|---|---|---|
| COCO val2017 | Object detection images | /mnt/forge-data/datasets/coco/val2017 | 788MB | READY |
| COCO DINOv2 features | Pre-computed ViT features | /mnt/forge-data/shared_infra/datasets/coco_dinov2_features/ | 9.9GB | READY |

## CUDA Kernels
| Kernel | File | Target | Status |
|---|---|---|---|
| svd_nuclear_prox | kernels/aoexp_cuda_kernels.cu | sm_89 (L4) | Compiled + verified |
| fused_mask_ce | kernels/aoexp_cuda_kernels.cu | sm_89 (L4) | Compiled + verified |

Saved to: `/mnt/forge-data/shared_infra/cuda_extensions/aoexp_nuclear_prox/`

## Exports
| Format | Path | Size |
|---|---|---|
| PyTorch (.pth) | /mnt/artifacts-datai/exports/DEF-aoexp/perturbation.pth | 4.7MB |
| SafeTensors | /mnt/artifacts-datai/exports/DEF-aoexp/perturbation.safetensors | 4.7MB |
| ONNX | /mnt/artifacts-datai/exports/DEF-aoexp/perturbation_applier.onnx | 4.7MB |
| TRT FP32 | /mnt/artifacts-datai/exports/DEF-aoexp/perturbation_applier_fp32.trt | 4.8MB |
| TRT FP16 | /mnt/artifacts-datai/exports/DEF-aoexp/perturbation_applier_fp16.trt | 4.8MB |
