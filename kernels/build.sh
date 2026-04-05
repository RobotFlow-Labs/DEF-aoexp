#!/bin/bash
# Build AO-Exp CUDA kernels for L4 (sm_89) with torch cu128
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building AO-Exp CUDA kernels ==="
echo "Target: sm_89 (L4), CUDA 12.8, torch cu128"

source /mnt/forge-data/activate.sh

# Build
uv run python setup.py build_ext --inplace 2>&1

# Verify
uv run python -c "import aoexp_cuda_kernels; print('OK: aoexp_cuda_kernels loaded')"

# Copy wheel to shared infra
DEST="/mnt/forge-data/shared_infra/cuda_extensions/aoexp_nuclear_prox"
mkdir -p "$DEST"
cp -v aoexp_cuda_kernels*.so "$DEST/" 2>/dev/null || true
cp -v aoexp_cuda_kernels.cu "$DEST/"
cp -v setup.py "$DEST/"

echo "=== Build complete ==="
