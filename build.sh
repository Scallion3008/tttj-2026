#!/usr/bin/env bash

# Create the locked Python environment and build the production FlashAttention-3
# wheel. This script is safe to rerun; it restores the vendor checkout on exit.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CUDA_ROOT=/usr/local/cuda-12.9
FA3_ROOT="${REPO_ROOT}/vendor/flash-attention"
FA3_PATCH="${REPO_ROOT}/patches/flash-attention/system-cuda-12.9.patch"
WHEEL_ROOT="${REPO_ROOT}/.artifacts/wheels"
FA3_COMMIT=ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820
CUTLASS_COMMIT=7127592069c2fe01b041e174ba4345ef9b279671

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

command -v git >/dev/null || fail "git is required"
command -v uv >/dev/null || fail "uv is required (https://docs.astral.sh/uv/)"
[[ -x "${CUDA_ROOT}/bin/nvcc" ]] || fail "CUDA 12.9 is required at ${CUDA_ROOT}"
"${CUDA_ROOT}/bin/nvcc" --version | grep -q 'release 12\.9' \
    || fail "${CUDA_ROOT}/bin/nvcc is not CUDA 12.9"

git -C "${REPO_ROOT}" submodule update --init -- vendor/flash-attention
git -C "${FA3_ROOT}" submodule update --init -- csrc/cutlass

[[ "$(git -C "${FA3_ROOT}" rev-parse HEAD)" == "${FA3_COMMIT}" ]] \
    || fail "FlashAttention checkout does not match the locked commit ${FA3_COMMIT}"
[[ "$(git -C "${FA3_ROOT}/csrc/cutlass" rev-parse HEAD)" == "${CUTLASS_COMMIT}" ]] \
    || fail "CUTLASS checkout does not match the locked commit ${CUTLASS_COMMIT}"

PATCH_APPLIED_BY_BUILD=FALSE
WHEEL_TMP=
cleanup() {
    if [[ "${PATCH_APPLIED_BY_BUILD}" == TRUE ]]; then
        git -C "${FA3_ROOT}" apply --reverse "${FA3_PATCH}"
    fi
    if [[ -n "${WHEEL_TMP}" ]]; then
        rm -rf -- "${WHEEL_TMP}"
    fi
}
trap cleanup EXIT

if git -C "${FA3_ROOT}" apply --check "${FA3_PATCH}" 2>/dev/null; then
    git -C "${FA3_ROOT}" apply "${FA3_PATCH}"
    PATCH_APPLIED_BY_BUILD=TRUE
elif ! git -C "${FA3_ROOT}" apply --reverse --check "${FA3_PATCH}" 2>/dev/null; then
    fail "the FlashAttention patch cannot be applied cleanly; inspect ${FA3_ROOT}"
fi

export CUDA_HOME="${CUDA_ROOT}"
export PATH="${CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MAX_JOBS="${MAX_JOBS:-4}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
export BUILD_TARGET=cuda
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_DISABLE_BACKWARD=TRUE
export FLASH_ATTENTION_DISABLE_SPLIT=FALSE
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

uv sync --directory "${REPO_ROOT}" --frozen

WHEEL_TMP="$(mktemp -d)"
(
    cd "${FA3_ROOT}/hopper"
    uv run --project "${REPO_ROOT}" --frozen python setup.py \
        bdist_wheel --dist-dir "${WHEEL_TMP}"
)

shopt -s nullglob
WHEELS=("${WHEEL_TMP}"/flash_attn_3-*.whl)
[[ "${#WHEELS[@]}" -eq 1 ]] \
    || fail "expected one FlashAttention-3 wheel, found ${#WHEELS[@]}"
mkdir -p "${WHEEL_ROOT}"
cp -- "${WHEELS[0]}" "${WHEEL_ROOT}/"
uv pip install --python "${REPO_ROOT}/.venv/bin/python" \
    --no-deps --reinstall "${WHEELS[0]}"

"${REPO_ROOT}/.venv/bin/python" -c \
    'from flash_attn_3.flash_attn_interface import flash_attn_func; print("FlashAttention-3 import: OK")'
echo "Built and installed ${WHEEL_ROOT}/$(basename -- "${WHEELS[0]}")"
