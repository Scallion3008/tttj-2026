#!/usr/bin/env bash
#SBATCH --job-name=tttj-step1
#SBATCH --output=job-scripts/run_step_1_h200-%j.out
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
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
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS="${SLURM_CPUS_PER_TASK}"
export CUDA_MODULE_LOADING=LAZY

TTTJ_EXTENSION_DIR="${SLURM_TMPDIR:-/tmp}/tttj-2026-step1-extensions-${SLURM_JOB_ID}"
mkdir -p "${TTTJ_EXTENSION_DIR}"
export TORCH_EXTENSIONS_DIR="${TTTJ_EXTENSION_DIR}"

hostname
nvcc --version
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

uv run --frozen python benchmark_steps_1_2.py --mode checkpoints --verbose-build
