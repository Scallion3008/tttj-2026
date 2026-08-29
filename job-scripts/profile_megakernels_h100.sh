#!/usr/bin/env bash
#SBATCH --job-name=tttj-ncu-h100
#SBATCH --output=job-scripts/profile_megakernels_h100-%j.out
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100-96:1

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

profile_sections=(
    --section SpeedOfLight
    --section LaunchStats
    --section Occupancy
    --section ComputeWorkloadAnalysis
    --section MemoryWorkloadAnalysis
    --section SchedulerStats
    --section WarpStateStats
)

for case_number in ${PROFILE_CASES:-5 6 1 9 10 2 3 4 12 7 11}; do
    if [[ "${case_number}" == "2" || "${case_number}" == "3" || \
          "${case_number}" == "4" || "${case_number}" == "12" ]]; then
        kernel_regex=_static_sequence_dag_megakernel
    else
        kernel_regex=_transformer_megakernel
    fi
    report="job-scripts/ncu_h100_case${case_number}_${SLURM_JOB_ID}"
    "${NCU}" \
        --force-overwrite \
        --target-processes all \
        --kernel-name-base function \
        --kernel-name "regex:${kernel_regex}" \
        --launch-count 1 \
        "${profile_sections[@]}" \
        --export "${report}" \
        uv run --frozen python profile_megakernel.py --case "${case_number}"
    "${NCU}" --import "${report}.ncu-rep" --page details > "${report}.txt"
done
