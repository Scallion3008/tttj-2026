#!/usr/bin/env bash
#SBATCH --job-name=tttj-ncu
#SBATCH --output=job-scripts/outputs/profile_megakernel_h200-%j.out
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

CUDA_ROOT=/usr/local/cuda-12.9
NCU="${CUDA_ROOT}/bin/ncu"
if [[ ! -x "${CUDA_ROOT}/bin/nvcc" || ! -x "${NCU}" ]]; then
    echo "ERROR: CUDA 12.9 and Nsight Compute are required under ${CUDA_ROOT}" >&2
    exit 1
fi

export CUDA_HOME="${CUDA_ROOT}"
export PATH="${CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_MODULE_LOADING=LAZY

hostname
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
"${NCU}" --version | tail -1

if [[ "${PROFILE_LAUNCH_ONLY:-0}" == "1" ]]; then
    profile_sections=(--section LaunchStats)
else
    profile_sections=(
        --set detailed
        --section InstructionStats
        --section SchedulerStats
        --section WarpStateStats
    )
fi

for case_number in ${PROFILE_CASES:-5 6}; do
    report="job-scripts/outputs/ncu_megakernel_case${case_number}_${SLURM_JOB_ID}"
    "${NCU}" \
        --force-overwrite \
        --target-processes all \
        --kernel-name-base function \
        --kernel-name regex:_transformer_megakernel \
        --launch-count 1 \
        "${profile_sections[@]}" \
        --export "${report}" \
        uv run --frozen python -m profiling.profile_megakernel \
            --case "${case_number}"
    "${NCU}" --import "${report}.ncu-rep" --page details > "${report}.txt"
done
