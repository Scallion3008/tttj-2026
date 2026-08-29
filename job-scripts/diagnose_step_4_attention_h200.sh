#!/usr/bin/env bash
#SBATCH --job-name=tttj-s4-attn
#SBATCH --output=job-scripts/outputs/diagnose_step_4_attention_h200-%j.out
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
nvcc --version
uv run --frozen python -m profiling.diagnose_step_4_attention "$@"
