#!/usr/bin/env bash
#SBATCH --job-name=tttj-close-gap
#SBATCH --output=job-scripts/outputs/close_compile_gap_h100-%j.out
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

# Retain one full H100 allocation through component tuning, paired latency,
# the broad accuracy matrix, and the public correctness regression.
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

uv run --frozen python -u -m profiling.tune_step_7_exact_scores

uv run --frozen python -u -m benchmarks.benchmark_step_7 \
    --allow-h100 \
    --attention-mask 0 \
    --providers hlinear-cudnn ilinear-cudnn alinear-cudnn production

uv run --frozen python -u -m benchmarks.benchmark_compile_comparison \
    --cases 13 \
    --modes max-autotune \
    --warmup 25 \
    --rep 100

uv run --frozen python -u -m benchmarks.benchmark_step_7 \
    --allow-h100 \
    --attention-mask 0 \
    --providers production \
    --accuracy-matrix \
    --accuracy-trials 3 \
    --matrix-provider production \
    --padding-ratios 0.0 0.25 \
    --scales 0.0001 0.1 1.0 10.0 1000.0 \
    --skip-latency

uv run --frozen python -u -m benchmarks.regression \
    --cases 13 \
    --trials 5 \
    --padding-ratios 0.0 0.25
