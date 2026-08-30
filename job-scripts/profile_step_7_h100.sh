#!/usr/bin/env bash
#SBATCH --job-name=tttj-step7-ncu
#SBATCH --output=job-scripts/outputs/profile_step_7_h100-%j.out
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

CUDA_ROOT=/usr/local/cuda-12.9
NCU="${CUDA_ROOT}/bin/ncu"
if [[ ! -x "${CUDA_ROOT}/bin/nvcc" || ! -x "${NCU}" ]]; then
    echo "ERROR: CUDA 12.9 and Nsight Compute are required under ${CUDA_ROOT}" >&2
    exit 1
fi
CUDNN_ROOT="${SLURM_SUBMIT_DIR}/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
export CUDA_HOME="${CUDA_ROOT}"
export PATH="${CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDNN_ROOT}:${CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_MODULE_LOADING=LAZY

hostname
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
"${NCU}" --version | tail -1

report="job-scripts/outputs/ncu_step7_exact_attention_${SLURM_JOB_ID}"
"${NCU}" \
    --force-overwrite \
    --target-processes all \
    --nvtx \
    --nvtx-include "step7_exact_attention/" \
    --section SpeedOfLight \
    --section LaunchStats \
    --section Occupancy \
    --section ComputeWorkloadAnalysis \
    --section MemoryWorkloadAnalysis \
    --export "${report}" \
    uv run --frozen python -m profiling.profile_step_7
"${NCU}" --import "${report}.ncu-rep" --page details > "${report}.txt"
