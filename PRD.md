# PRD: DEF-aoexp — Structured Universal Adversarial Attacks on Object Detection

## Overview
Implementation of the AO-Exp (Adaptive Optimistic Exponentiated Gradient) attack framework
for generating universal adversarial perturbations against object detectors on video sequences.

## Paper Reference
- **Title**: Structured Universal Adversarial Attacks on Object Detection for Video Sequences
- **ArXiv**: 2510.14460
- **Authors**: Sven Jacob, Weijia Shao, Gjergji Kasneci
- **Conference**: GCPR 2025

## Core Components

### 1. AO-Exp Optimizer (`ao_exp_optimizer.py`)
- Per-channel SVD decomposition of perturbation
- Exponentiated gradient mirror descent
- Lambert W function proximal operator for nuclear + Frobenius norm
- Top-k singular value truncation
- Adaptive learning rate via accumulated singular value norms

### 2. Attack Engine (`attack_engine.py`)
- Mask R-CNN target detector (torchvision pretrained)
- Foreground/background mask cross-entropy loss
- Confidence penalty loss
- Batch gradient accumulation across video frames
- IoU-based evaluation metrics

### 3. Data Pipeline (`data_pipeline.py`)
- COCO dataset loader (adapted from original video surveillance datasets)
- Pre-computed DINOv2 feature integration from shared cache
- Frame batching with configurable batch size

### 4. CUDA Kernels (`kernels/`)
- `svd_nuclear_prox`: Fused SVD update + nuclear norm proximal operator
- `mask_ce_loss`: Fused foreground/background mask CE loss computation

### 5. Export Pipeline
- PyTorch (.pth) + SafeTensors
- ONNX export
- TensorRT FP16 + FP32 (mandatory)

## Datasets
- COCO 2017 (val): /mnt/forge-data/datasets/coco/
- COCO DINOv2 features: /mnt/forge-data/shared_infra/datasets/coco_dinov2_features/

## Success Criteria
- Generate universal perturbation that reduces detection mAP by >30%
- Nuclear norm of perturbation < threshold (imperceptibility)
- Training completes within 60-70% VRAM on single L4
- All export formats produced successfully
