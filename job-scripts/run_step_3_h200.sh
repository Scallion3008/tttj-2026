#!/usr/bin/env bash
#SBATCH --job-name=tttj-step3
#SBATCH --output=job-scripts/outputs/run_step_3_h200-%j.out
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
export CUDA_MODULE_LOADING=LAZY

hostname
nvcc --version
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

benchmark_args=()
if [[ "${BENCH_QUICK:-0}" == "1" ]]; then
    benchmark_args+=(--quick)
fi
uv run --frozen python -m benchmarks.benchmark_step_3 \
    "${benchmark_args[@]}" "$@"
