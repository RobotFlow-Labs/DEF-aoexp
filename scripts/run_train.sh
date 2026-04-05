#!/bin/bash
# Launch AO-Exp CUDA training with nohup+disown
# Target: 60-70% VRAM on single L4 (23GB → ~14-16GB)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$MODULE_DIR"

source /mnt/forge-data/activate.sh

# Default GPU
GPU="${1:-0}"
CONFIG="${2:-configs/default.toml}"

export CUDA_VISIBLE_DEVICES="$GPU"

echo "=== AO-Exp Training ==="
echo "GPU: $GPU"
echo "Config: $CONFIG"
echo "Log: /mnt/artifacts-datai/logs/DEF-aoexp/train_cu.log"

# Create output dirs
mkdir -p /mnt/artifacts-datai/checkpoints/DEF-aoexp
mkdir -p /mnt/artifacts-datai/logs/DEF-aoexp
mkdir -p /mnt/artifacts-datai/exports/DEF-aoexp

# Launch with nohup + disown
nohup uv run python -m def_aoexp.train_cu --config "$CONFIG" \
    > /mnt/artifacts-datai/logs/DEF-aoexp/train_cu_stdout.log 2>&1 &
disown

PID=$!
echo "Training launched: PID=$PID"
echo "Monitor: tail -f /mnt/artifacts-datai/logs/DEF-aoexp/train_cu.log"
echo "GPU: watch -n2 nvidia-smi"
