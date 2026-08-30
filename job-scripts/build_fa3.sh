#!/usr/bin/env bash
#SBATCH --job-name=tttj-fa3-build
#SBATCH --output=job-scripts/outputs/build_fa3-%j.out
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=normal

# FA3 is compiled for sm_90 by nvcc and does not need an allocated GPU.  The
# resulting wheel is shared by H100 and H200 benchmark jobs.
set -euo pipefail

CUDA_ROOT=/usr/local/cuda-12.9
SOURCE_ROOT=/home/e/e1525944/.cache/flash-attention-step7
WHEEL_ROOT=/home/e/e1525944/.cache/flash-attention-step7-wheels
if [[ ! -x "${CUDA_ROOT}/bin/nvcc" ]]; then
    echo "ERROR: CUDA 12.9 is required at ${CUDA_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${SOURCE_ROOT}/hopper/setup.py" ]]; then
    echo "ERROR: FlashAttention source is missing at ${SOURCE_ROOT}" >&2
    exit 1
fi
if ! grep -q "if False and bare_metal_version" "${SOURCE_ROOT}/hopper/setup.py"; then
    patch --directory="${SOURCE_ROOT}" --strip=1 \
        < "${SLURM_SUBMIT_DIR}/profiling/fa3_system_cuda_12_9.patch"
fi

export CUDA_HOME="${CUDA_ROOT}"
export PATH="${CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MAX_JOBS=4
export NVCC_THREADS=2
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_DISABLE_BACKWARD=TRUE
export FLASH_ATTENTION_DISABLE_SPLIT=TRUE
export FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE
export FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
export FLASH_ATTENTION_DISABLE_LOCAL=TRUE
export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
export FLASH_ATTENTION_DISABLE_PACKGQA=TRUE
export FLASH_ATTENTION_DISABLE_FP8=TRUE
export FLASH_ATTENTION_DISABLE_VARLEN=TRUE
export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
export FLASH_ATTENTION_DISABLE_HDIM128=TRUE
export FLASH_ATTENTION_DISABLE_HDIM192=TRUE
export FLASH_ATTENTION_DISABLE_HDIM256=TRUE
export FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE
export FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE
export FLASH_ATTENTION_DISABLE_SM80=TRUE

hostname
nvcc --version
mkdir -p "${WHEEL_ROOT}"
cd "${SOURCE_ROOT}/hopper"
uv run --project "${SLURM_SUBMIT_DIR}" --frozen python setup.py \
    bdist_wheel --dist-dir "${WHEEL_ROOT}"
uv pip install --python "${SLURM_SUBMIT_DIR}/.venv/bin/python" --no-deps \
    "${WHEEL_ROOT}"/flash_attn_3-*.whl
