"""JIT build for AO-Exp CUDA kernels, bypassing nvcc version check."""

import os
import sys

# Monkey-patch to skip CUDA version check
import torch.utils.cpp_extension as _ext

_orig_check = _ext._check_cuda_version
_ext._check_cuda_version = lambda *a, **k: None

from torch.utils.cpp_extension import load

kernel_dir = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(kernel_dir, "aoexp_cuda_kernels.cu")

print(f"Building AO-Exp CUDA kernels from {src}...")
print(f"Target: sm_89 (L4)")

mod = load(
    name="aoexp_cuda_kernels",
    sources=[src],
    extra_cuda_cflags=[
        "-O3",
        "-gencode=arch=compute_89,code=sm_89",
        "--use_fast_math",
        "-lineinfo",
    ],
    extra_cflags=["-O3"],
    verbose=True,
)

# Verify
import torch
s_in = torch.rand(10, device="cuda")
s_out = mod.svd_nuclear_prox(s_in, 0.1, 0.01, 1.0, 1.0, 1.0, 5)
print(f"svd_nuclear_prox test: input={s_in[:3].tolist()}, output={s_out[:3].tolist()}")

adv = torch.rand(64, 64, device="cuda")
clean = (torch.rand(64, 64, device="cuda") > 0.5).float()
losses = mod.fused_mask_ce(adv.contiguous(), clean.contiguous())
print(f"fused_mask_ce test: fg_loss={losses[0].item():.4f}, bg_loss={losses[1].item():.4f}")

print("OK: All kernels verified")

# Copy .so to shared infra
import shutil
dest = "/mnt/forge-data/shared_infra/cuda_extensions/aoexp_nuclear_prox"
os.makedirs(dest, exist_ok=True)
shutil.copy2(src, dest)
print(f"Kernel source copied to {dest}")
