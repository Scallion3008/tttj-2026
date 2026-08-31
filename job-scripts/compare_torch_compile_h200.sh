#!/usr/bin/env bash
#SBATCH --job-name=tttj-compile-h200
#SBATCH --output=job-scripts/outputs/compare_torch_compile_h200-%j.out
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

CUDA_ROOT=/usr/local/cuda-12.9
if [[ ! -x "${CUDA_ROOT}/bin/nvcc" ]]; then
    echo "ERROR: CUDA 12.9 is required at ${CUDA_ROOT}" >&2
    exit 1
fi
CUDNN_ROOT="${SLURM_SUBMIT_DIR}/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
if [[ ! -f "${CUDNN_ROOT}/libcudnn_engines_runtime_compiled.so.9" ]]; then
    echo "ERROR: the complete pinned cuDNN wheel is missing at ${CUDNN_ROOT}" >&2
    exit 1
fi

export CUDA_HOME="${CUDA_ROOT}"
export PATH="${CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDNN_ROOT}:${CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_MODULE_LOADING=LAZY

hostname
nvcc --version
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

uv run --frozen python -u -m benchmarks.benchmark_compile_comparison \
    --cases 1 2 3 4 5 6 7 8 9 10 11 12 13 14 \
    --modes default reduce-overhead max-autotune \
    --warmup 25 --rep 100 \
    --long-context-warmup 3000 --long-context-rep 15000
