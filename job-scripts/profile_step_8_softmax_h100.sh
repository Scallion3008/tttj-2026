#!/usr/bin/env bash
#SBATCH --job-name=tttj-step8-softmax-ncu
#SBATCH --output=job-scripts/outputs/profile_step_8_softmax_h100-%j.out
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
export CUDA_HOME="${CUDA_ROOT}"
export PATH="${SLURM_SUBMIT_DIR}/.venv/bin:${CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

hostname
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
REPORT="${SLURM_SUBMIT_DIR}/job-scripts/outputs/ncu_step8_softmax_${SLURM_JOB_ID}"
"${NCU}" \
    --profile-from-start off \
    --kernel-name "regex:.*exact_softmax_kernel.*" \
    --launch-count 1 \
    --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,smsp__sass_thread_inst_executed_op_special_pred_on.sum,smsp__warps_active.avg.pct_of_peak_sustained_active \
    --export "${REPORT}" \
    "${SLURM_SUBMIT_DIR}/.venv/bin/python" -m profiling.profile_step_8 \
        --mode softmax --rows 8192 --sequence 100000 --ncu
"${NCU}" --import "${REPORT}.ncu-rep" --page raw \
    > "${REPORT}.txt"
