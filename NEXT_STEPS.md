# NEXT_STEPS — DEF-aoexp
## Last Updated: 2026-04-05
## Status: BUILD_COMPLETE
## MVP Readiness: 95%

### Completed
- [x] Reference repo cloned (repositories/AO-Exp-Attack)
- [x] Paper analysis (arXiv:2510.14460, GCPR 2025)
- [x] Python scaffold (pyproject.toml, src/def_aoexp/, configs/, tests/)
- [x] Core AO-Exp optimizer (GPU PyTorch, Lambert W, per-channel SVD)
- [x] Attack engine (Mask R-CNN, foreground/background mask CE, batch grad)
- [x] Data pipeline (COCO val2017, DINOv2 feature cache)
- [x] CUDA kernels (2 ops: svd_nuclear_prox, fused_mask_ce) — sm_89
- [x] Training completed (50 iters, 100 images, 660s)
- [x] Full export: pth + safetensors + ONNX + TRT FP16 + TRT FP32
- [x] Pushed to HuggingFace: ilessio-aiflowlab/DEF-aoexp
- [x] Code review: ruff clean, 24/24 tests passing
- [x] TRAINING_REPORT.md generated
- [x] Docker serving infrastructure
- [x] Git pushed (8 commits to main)

### Completed (Phase 2)
- [x] CUDA kernel JIT integration verified (svd_nuclear_prox + fused_mask_ce)
- [x] Kernel benchmark: svd_prox 78x speedup, mask_ce 4.9x speedup
- [x] UAV evaluation pipeline (SERAPHIM, DroneVehicle-night, BirdDrone)
- [x] UAV eval results: 0% detection drop (perturbation near-zero, regularization-dominated)
- [x] TRAINING_REPORT.md updated with benchmarks + UAV eval

### Remaining
- [ ] Re-train with reduced regularization (λ₁=0.01, λ₂=0.001) for stronger attack
- [ ] Implement randomized SVD to reduce per-step cost (currently 357ms dominated by SVD)
- [ ] Full-scale UAV evaluation (all images, not just 200 sample)
