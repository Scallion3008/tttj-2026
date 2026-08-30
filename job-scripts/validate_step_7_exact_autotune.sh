#!/usr/bin/env bash
#SBATCH --job-name=tttj-step7-exact-auto
#SBATCH --output=job-scripts/outputs/validate_step_7_exact_autotune-%j.out
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

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
export CUDA_MODULE_LOADING=LAZY

hostname
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

uv run --frozen python -u -m benchmarks.benchmark_step_7 \
    --allow-h100 \
    --attention-mask 0 \
    --providers production \
    --accuracy-matrix \
    --accuracy-trials 3 \
    --matrix-provider production \
    --padding-ratios 0.0 0.25 \
    --scales 0.0001 0.1 1.0 10.0 1000.0

uv run --frozen python -u -m benchmarks.regression \
    --cases 13 \
    --trials 5 \
    --padding-ratios 0.0 0.25
