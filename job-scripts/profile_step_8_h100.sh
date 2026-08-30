#!/usr/bin/env bash
#SBATCH --job-name=tttj-step8-ncu
#SBATCH --output=job-scripts/outputs/profile_step_8_h100-%j.out
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

CUDA_ROOT=/usr/local/cuda-12.9
NCU="${CUDA_ROOT}/bin/ncu"
if [[ ! -x "${CUDA_ROOT}/bin/nvcc" || ! -x "${NCU}" ]]; then
    echo "ERROR: CUDA 12.9 and Nsight Compute are required at ${CUDA_ROOT}" >&2
    exit 1
fi
CUDNN_ROOT="${SLURM_SUBMIT_DIR}/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
export CUDA_HOME="${CUDA_ROOT}"
export PATH="${SLURM_SUBMIT_DIR}/.venv/bin:${CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDNN_ROOT}:${CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

hostname
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
REPORT="${SLURM_SUBMIT_DIR}/job-scripts/outputs/ncu_step8_attention_${SLURM_JOB_ID}"
"${NCU}" \
    --profile-from-start off \
    --kernel-name "regex:^device_kernel$" \
    --launch-count 1 \
    --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active \
    --export "${REPORT}" \
    "${SLURM_SUBMIT_DIR}/.venv/bin/python" -m profiling.profile_step_8 \
        --mode attention --ncu
"${NCU}" --import "${REPORT}.ncu-rep" --page raw \
    > "${REPORT}.txt"
