#!/usr/bin/env bash
#SBATCH --job-name=tttj-step8-query-sweep
#SBATCH --output=job-scripts/outputs/tune_step_8_query_chunks_h200-%j.out
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

CUDA_ROOT=/usr/local/cuda-12.9
if [[ ! -x "${CUDA_ROOT}/bin/nvcc" ]]; then
    echo "ERROR: CUDA 12.9 is required at ${CUDA_ROOT}" >&2
    exit 1
fi
CUDNN_ROOT="${SLURM_SUBMIT_DIR}/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
export CUDA_HOME="${CUDA_ROOT}"
export PATH="${CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDNN_ROOT}:${CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

hostname
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
if (( $# )); then
    query_chunks=("$@")
else
    query_chunks=(384 448 512 576 640 768 896)
fi
for query_chunk in "${query_chunks[@]}"; do
    echo "query_chunk=${query_chunk}"
    TTTJ_STEP8_EXACT_QUERY_CHUNK="${query_chunk}" \
        uv run --frozen python -m benchmarks.benchmark_step_8 \
            --provider production --skip-accuracy --warmup 1 --repetitions 2
done
