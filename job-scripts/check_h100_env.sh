#!/usr/bin/env bash
#SBATCH --job-name=tttj-env
#SBATCH --output=job-scripts/check_h100_env-%j.out
#SBATCH --time=00:05:00

set -euo pipefail

CUDA_ROOT=/usr/local/cuda-12.9
if [[ ! -x "${CUDA_ROOT}/bin/nvcc" ]]; then
    echo "ERROR: CUDA 12.9 is required at ${CUDA_ROOT}" >&2
    exit 1
fi

export PATH="${CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

hostname
nvcc --version
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
uv run --frozen python --version
uv run --frozen python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"device={props.name}")
    print(f"sm_count={props.multi_processor_count}")
    print(f"total_memory={props.total_memory}")
PY
