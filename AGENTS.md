# Instructions

This repo seeks to optimize the forward pass of a simple transformer model by implementing it as a collection of megakernels. Refer to [README.md](README.md) for details, including numerical accuracy constraints and implementation details.

## Environment

If the hostname begins with `xlogin` or `xcnc`, you are working on a Slurm login node. Submit jobs to run on H100-96 GPUs using `sbatch --gres=gpu:h100-96:1 --partition=gpu --nodes=1 --nodelist=xgpi[0-9] <script path>`.

H100-96 compute nodes should have CUDA installed at `/usr/local/cuda-12.9/`. If CUDA 12.9 is not available, fail immediately instead of falling back to another CUDA version.

## Repo structure

- `llm-scratchpad`: notes left by previous agent sessions for consumption by other agents.
- `job-scripts`: Slurm job scripts
