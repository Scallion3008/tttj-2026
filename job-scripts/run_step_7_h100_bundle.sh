#!/usr/bin/env bash
#SBATCH --job-name=tttj-step7-h100
#SBATCH --output=job-scripts/outputs/run_step_7_h100_bundle-%j.out
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Consolidate all remaining full-H100 work so an acquired H100-96 slot is not
# released between the candidate benchmark, compiler diagnostic, and public
# correctness regression.
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
    --allow-h100 \
    --attention-mask 0x0 \
    --providers \
    torch \
    compiled-cudnn \
    clinear-cudnn \
    hlinear-cudnn \
    graph-compiled-cudnn \
    graph-clinear-cudnn \
    graph-hlinear-cudnn \
    production \
    --microbench

uv run --frozen python -m profiling.diagnose_step_7_compiled_attention

uv run --frozen python -m benchmarks.regression \
    --cases 13 \
    --trials 3 \
    --padding-ratios 0.0 0.25
