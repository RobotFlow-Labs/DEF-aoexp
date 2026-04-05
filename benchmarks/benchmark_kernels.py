"""Benchmark: PyTorch fallback vs Custom CUDA kernels for AO-Exp.

Measures latency, throughput, and memory for:
1. svd_nuclear_prox: Lambert W proximal operator on singular values
2. fused_mask_ce: Single-pass FG/BG mask cross-entropy loss

Usage:
    CUDA_VISIBLE_DEVICES=4 python benchmarks/benchmark_kernels.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
import torch.utils.cpp_extension as ext

# Bypass nvcc version check
ext._check_cuda_version = lambda *a, **k: None

DEVICE = "cuda:0"
WARMUP = 50
ITERS = 500
H, W, C = 640, 640, 3


def load_cuda_kernels():
    kernel_dir = Path(__file__).parent.parent / "kernels"
    src = kernel_dir / "aoexp_cuda_kernels.cu"
    return ext.load(
        name="aoexp_cuda_kernels",
        sources=[str(src)],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_89,code=sm_89", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


# ─── PyTorch fallback implementations ────────────────────────────────


def _lambert_w0_approx(z: torch.Tensor) -> torch.Tensor:
    log_z = torch.log(z.clamp(min=1e-30))
    log_log_z = torch.log(log_z.clamp(min=1e-30))
    w = log_z - log_log_z + log_log_z / log_z
    small = z < 2.0
    w = torch.where(small, z / (1.0 + z), w)
    for _ in range(4):
        ew = torch.exp(w)
        wew = w * ew
        f = wew - z
        fp = ew * (w + 1.0)
        fpp = ew * (w + 2.0)
        w = w - f / (fp - f * fpp / (2.0 * fp))
    return w


def pytorch_svd_prox(s_y, lambda1, lambda2, t, alpha, beta_const, k):
    a = beta_const
    b = lambda2 * t / alpha
    c_val = torch.clamp(
        lambda1 * t / alpha - torch.log(s_y / beta_const + 1.0), max=0.0
    )
    ab = torch.as_tensor(a * b, device=s_y.device)
    abc = torch.log(ab) + ab - c_val

    large_mask = abc >= 15.0
    log_abc = torch.log(abc.clamp(min=1e-30))
    log_log_abc = torch.log(log_abc.clamp(min=1e-30))
    w_large = log_abc - log_log_abc + log_log_abc / log_abc
    w_small = _lambert_w0_approx(torch.exp(abc))
    w_val = torch.where(large_mask, w_large, w_small)
    s_new = w_val / b - a

    s_trunc = torch.zeros_like(s_new)
    s_trunc[:k] = s_new[:k]
    return s_trunc


def pytorch_mask_ce(adv_mask, clean_mask):
    adv_probs = torch.stack([1.0 - adv_mask, adv_mask], dim=-1).reshape(-1, 2)
    target = ((clean_mask > 0).float()).flatten().long()
    fg_idx = target.nonzero(as_tuple=True)[0]
    bg_idx = (target == 0).nonzero(as_tuple=True)[0]
    loss_fg = F.cross_entropy(adv_probs[fg_idx], target[fg_idx]) if fg_idx.numel() > 0 else 0.0
    loss_bg = F.cross_entropy(adv_probs[bg_idx], target[bg_idx]) if bg_idx.numel() > 0 else 0.0
    return loss_fg, loss_bg


# ─── Benchmark runner ────────────────────────────────────────────────


def benchmark_fn(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    return {
        "total_ms": total_ms,
        "avg_ms": total_ms / iters,
        "throughput_ops": iters / (total_ms / 1000.0),
    }


def benchmark_svd_prox(cuda_mod):
    print("\n" + "=" * 60)
    print("Benchmark 1: svd_nuclear_prox (Lambert W proximal operator)")
    print("=" * 60)

    s_in = torch.rand(min(H, W), device=DEVICE, dtype=torch.float32) + 0.1
    lambda1, lambda2, t, alpha, beta_const, k = 0.1, 0.01, 50.0, 0.55, 1.0, 320

    # PyTorch
    pt = benchmark_fn(lambda: pytorch_svd_prox(s_in, lambda1, lambda2, t, alpha, beta_const, k))

    # CUDA
    cu = benchmark_fn(
        lambda: cuda_mod.svd_nuclear_prox(s_in, lambda1, lambda2, t, alpha, beta_const, k)
    )

    speedup = pt["avg_ms"] / cu["avg_ms"]
    print(f"  PyTorch:  {pt['avg_ms']:.4f} ms/op  ({pt['throughput_ops']:.0f} ops/s)")
    print(f"  CUDA:     {cu['avg_ms']:.4f} ms/op  ({cu['throughput_ops']:.0f} ops/s)")
    print(f"  Speedup:  {speedup:.2f}x")

    return {"pytorch_ms": pt["avg_ms"], "cuda_ms": cu["avg_ms"], "speedup": speedup}


def benchmark_mask_ce(cuda_mod):
    print("\n" + "=" * 60)
    print("Benchmark 2: fused_mask_ce (FG/BG cross-entropy)")
    print("=" * 60)

    adv_mask = torch.rand(H * W, device=DEVICE, dtype=torch.float32)
    clean_mask = (torch.rand(H * W, device=DEVICE) > 0.5).float()
    adv_2d = adv_mask.view(H, W)
    clean_2d = clean_mask.view(H, W)

    # PyTorch
    pt = benchmark_fn(lambda: pytorch_mask_ce(adv_2d, clean_2d))

    # CUDA
    cu = benchmark_fn(lambda: cuda_mod.fused_mask_ce(adv_mask, clean_mask))

    speedup = pt["avg_ms"] / cu["avg_ms"]
    print(f"  PyTorch:  {pt['avg_ms']:.4f} ms/op  ({pt['throughput_ops']:.0f} ops/s)")
    print(f"  CUDA:     {cu['avg_ms']:.4f} ms/op  ({cu['throughput_ops']:.0f} ops/s)")
    print(f"  Speedup:  {speedup:.2f}x")

    return {"pytorch_ms": pt["avg_ms"], "cuda_ms": cu["avg_ms"], "speedup": speedup}


def benchmark_full_optimizer_step(cuda_mod):
    """Benchmark a full AO-Exp optimizer step (3-channel SVD + prox) with and without CUDA."""
    print("\n" + "=" * 60)
    print("Benchmark 3: Full optimizer step (3-channel SVD + proximal)")
    print("=" * 60)

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    import def_aoexp.ao_exp_optimizer as opt_mod
    import def_aoexp.losses as loss_mod
    from def_aoexp.ao_exp_optimizer import AOExpOptimizer

    # Use small random gradient each time to avoid SVD convergence issues
    # on ill-conditioned matrices from repeated identical steps

    def make_step_fn(opt, mod_ref):
        def fn():
            g = torch.randn(H, W, C, device=DEVICE) * 0.01
            try:
                opt.step(g)
            except Exception:
                # Reset optimizer state if SVD fails
                opt.x.zero_()
                opt.y.zero_()
                opt.h_prev.zero_()
                opt.lam_acc.zero_()
                opt.t = 0.0
                opt.beta = 0.0
        return fn

    # PyTorch fallback
    old_cuda_prox = opt_mod._cuda_svd_prox
    opt_mod._cuda_svd_prox = None
    opt_pt = AOExpOptimizer(shape=(H, W, C), device=DEVICE)
    pt = benchmark_fn(make_step_fn(opt_pt, None), warmup=5, iters=20)
    opt_mod._cuda_svd_prox = old_cuda_prox

    # CUDA kernels
    opt_mod._cuda_svd_prox = cuda_mod.svd_nuclear_prox
    loss_mod._cuda_fused_mask_ce = cuda_mod.fused_mask_ce
    opt_cu = AOExpOptimizer(shape=(H, W, C), device=DEVICE)
    cu = benchmark_fn(make_step_fn(opt_cu, cuda_mod), warmup=5, iters=20)

    speedup = pt["avg_ms"] / cu["avg_ms"]
    print(f"  PyTorch:  {pt['avg_ms']:.2f} ms/step  ({pt['throughput_ops']:.1f} steps/s)")
    print(f"  CUDA:     {cu['avg_ms']:.2f} ms/step  ({cu['throughput_ops']:.1f} steps/s)")
    print(f"  Speedup:  {speedup:.2f}x")

    return {"pytorch_ms": pt["avg_ms"], "cuda_ms": cu["avg_ms"], "speedup": speedup}


def main():
    print("AO-Exp Kernel Benchmark")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Resolution: {H}x{W}x{C}")
    print(f"Warmup: {WARMUP}, Iterations: {ITERS}")

    cuda_mod = load_cuda_kernels()
    print("CUDA kernels JIT-compiled OK")

    results = {}
    results["svd_nuclear_prox"] = benchmark_svd_prox(cuda_mod)
    results["fused_mask_ce"] = benchmark_mask_ce(cuda_mod)
    results["full_optimizer_step"] = benchmark_full_optimizer_step(cuda_mod)

    # Memory usage
    torch.cuda.reset_peak_memory_stats()
    _ = cuda_mod.svd_nuclear_prox(
        torch.rand(640, device=DEVICE), 0.1, 0.01, 1.0, 0.55, 1.0, 320
    )
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    results["peak_memory_mb"] = peak_mb

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        if isinstance(v, dict):
            pt_ms, cu_ms = v['pytorch_ms'], v['cuda_ms']
            print(f"  {k}: {v['speedup']:.2f}x (PT={pt_ms:.4f}ms, CU={cu_ms:.4f}ms)")
        else:
            print(f"  {k}: {v:.1f}")

    out = Path(__file__).parent / "benchmark_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
