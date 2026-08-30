#!/usr/bin/env bash
#SBATCH --job-name=tttj-step8-softmax
#SBATCH --output=job-scripts/outputs/validate_step_8_softmax-%j.out
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

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
uv run --frozen python -m profiling.validate_step_8_softmax
