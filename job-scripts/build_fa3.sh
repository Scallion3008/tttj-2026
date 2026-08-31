#!/usr/bin/env bash
#SBATCH --job-name=tttj-fa3-build
#SBATCH --output=job-scripts/outputs/build_fa3-%j.out
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=normal

# FA3 is compiled for sm_90 by nvcc and does not need an allocated GPU. The
# root build script owns dependency initialization, patching, and installation.
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
exec "${REPO_ROOT}/build.sh"
