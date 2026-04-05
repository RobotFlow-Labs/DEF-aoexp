"""Build script for AO-Exp CUDA kernels. Target: sm_89 (L4)."""

import os

# Skip CUDA version check — torch cu130 bundles its own CUDA runtime
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="aoexp_cuda_kernels",
    ext_modules=[
        CUDAExtension(
            "aoexp_cuda_kernels",
            ["aoexp_cuda_kernels.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-gencode=arch=compute_89,code=sm_89",
                    "--use_fast_math",
                    "-lineinfo",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
