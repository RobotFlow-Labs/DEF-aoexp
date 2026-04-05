# DEF-aoexp — ANIMA Defense Module (Wave 8)

## Paper
- **Title**: Structured Universal Adversarial Attacks on Object Detection for Video Sequences
- **ArXiv**: 2510.14460
- **Repo**: https://github.com/jsve96/AO-Exp-Attack
- **Domain**: Adversarial Attack / Object Detection
- **Conference**: GCPR 2025
- **Wave**: 8 (Defense-Only)

## Method Summary
AO-Exp generates **universal adversarial perturbations** (UAP) for video object detection using:
1. **Nuclear norm regularization** to promote structured, low-rank perturbations in the background
2. **Adaptive Optimistic Exponentiated Gradient** (AO-Exp) optimization with per-channel SVD
3. **Lambert W function** proximal operator for the nuclear+Frobenius norm regularizer
4. **Foreground/background mask CE loss** + confidence penalty from Mask R-CNN predictions

Key: single perturbation delta applied to ALL frames. Optimized via SVD decomposition per channel,
top-k singular value truncation, and exponentiated gradient mirror descent.

## YOU ARE THE BUILD AGENT

Your job:
1. Read CLAUDE.md first — always
2. Build with CUDA (torch cu128, sm_89 for L4)
3. Follow /anima-optimize-cuda-pipeline direction: custom kernels, fused ops
4. Every kernel optimization = IP we own

## Kernel Optimization IP
All modules MUST pursue custom kernel development:
- Fused SVD update + proximal operator kernel
- Nuclear norm computation kernel
- Memory-optimized batch gradient accumulation
- Benchmark: latency, throughput, memory vs baseline
- Store kernels in `kernels/` directory
- Store benchmarks in `benchmarks/` directory

## Conventions
- Package manager: `uv` (never pip)
- Python >= 3.10
- Build: hatchling
- Git prefix: [DEF-aoexp] per commit

## Reference Implementation
Source code at: repositories/AO-Exp-Attack/
Read ALL files in repositories/AO-Exp-Attack/ before building.
