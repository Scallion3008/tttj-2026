#!/usr/bin/env bash
#SBATCH --job-name=tttj-step8-fused-tile
#SBATCH --output=job-scripts/outputs/profile_step_8_fused_tile_h200-%j.out
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

CUDA_ROOT=/usr/local/cuda-12.9
if [[ ! -x "${CUDA_ROOT}/bin/nvcc" ]]; then
    echo "ERROR: CUDA 12.9 is required at ${CUDA_ROOT}" >&2
    exit 1
fi
export CUDA_HOME="${CUDA_ROOT}"
export PATH="${CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

hostname
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
uv run --frozen python -m profiling.profile_step_8 \
    --mode fused-tile --batch 32 --rows "${1:-512}" --sequence 100000
