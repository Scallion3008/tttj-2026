#!/usr/bin/env bash
#SBATCH --job-name=tttj-step7-h200
#SBATCH --output=job-scripts/outputs/run_step_7_h200_bundle-%j.out
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Run every authoritative H200 comparison in one allocation: exact-path and
# fusion tuning, official FA3 versus cuDNN, then the public regression.
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

uv run --frozen python -m benchmarks.benchmark_step_7 \
    --attention-mask 0x0 \
    --providers \
    torch \
    packed-cudnn \
    fused-cudnn \
    compiled-cudnn \
    clinear-cudnn \
    hlinear-cudnn \
    graph-compiled-cudnn \
    graph-clinear-cudnn \
    graph-hlinear-cudnn \
    production \
    --microbench \
    --accuracy-matrix \
    --matrix-provider production \
    --accuracy-trials 2 \
    --padding-ratios 0.0 0.25 \
    --scales 0.0001 0.1 1 10 1000

uv run --frozen python -m benchmarks.benchmark_step_7 \
    --quick \
    --attention-mask 0xf \
    --providers \
    torch \
    packed-fa3 \
    packed-cudnn \
    fused-fa3 \
    fused-cudnn \
    linear-fa3 \
    linear-cudnn \
    hlinear-cudnn \
    graph-fused-fa3 \
    graph-fused-cudnn \
    graph-hlinear-cudnn

uv run --frozen python -m benchmarks.regression \
    --cases 13 \
    --trials 3 \
    --padding-ratios 0.0 0.25
