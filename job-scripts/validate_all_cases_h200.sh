#!/usr/bin/env bash
#SBATCH --job-name=tttj-all-cases
#SBATCH --output=job-scripts/outputs/validate_all_cases_h200-%j.out
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

CUDA_ROOT=/usr/local/cuda-12.9
if [[ ! -x "${CUDA_ROOT}/bin/nvcc" ]]; then
    echo "ERROR: CUDA 12.9 is required at ${CUDA_ROOT}" >&2
    exit 1
fi

export CUDA_HOME="${CUDA_ROOT}"
export PATH="${CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_MODULE_LOADING=LAZY

hostname
nvcc --version
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

uv run --frozen python -u -m benchmarks.regression \
    --cases 1 2 3 4 5 6 7 8 9 10 11 12 13 14
uv run --frozen python -u -m benchmarks.benchmark_megakernels \
    --cases 1 2 3 4 5 6 7 8 9 10 11 12 13 14 \
    --warmup 25 --rep 100 \
    --long-context-warmup 3000 --long-context-rep 15000
