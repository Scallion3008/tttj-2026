#!/usr/bin/env bash
#SBATCH --job-name=tttj-step8-components
#SBATCH --output=job-scripts/outputs/profile_step_8_components_h200-%j.out
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

CUDA_ROOT=/usr/local/cuda-12.9
if [[ ! -x "${CUDA_ROOT}/bin/nvcc" ]]; then
    echo "ERROR: CUDA 12.9 is required at ${CUDA_ROOT}" >&2
    exit 1
fi
CUDNN_ROOT="${SLURM_SUBMIT_DIR}/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
export CUDA_HOME="${CUDA_ROOT}"
export PATH="${CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDNN_ROOT}:${CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

hostname
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
uv run --frozen python -m profiling.profile_step_8 --mode components
