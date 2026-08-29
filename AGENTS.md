# Instructions

This repo seeks to optimize the forward pass of a simple transformer model by implementing it as a collection of megakernels. Refer to [README.md](README.md) for details, including numerical accuracy constraints and implementation details.

## Environment

If the hostname begins with `xlogin` or `xcnc`, you are working on a Slurm login node. Submit jobs to run on H200-141 GPUs using `sbatch --gres=gpu:h200-141:1 <script path>`.

H200 compute node `xgpk0` should have CUDA installed at `/usr/local/cuda-12.9/`. If CUDA 12.9 is not available, fail immediately instead of falling back to another CUDA version.

## Repo structure

- `llm-scratchpad`: notes left by previous agent sessions for consumption by other agents.
- `job-scripts`: Slurm job scripts

## Optimization methodology

First work out the rough kernel structure (plans for each test case are detailed in `llm-scratchpad/initial_notes.md`), then hunt for potential algorithmic optimizations, and iterate until max possible speedup is achieved. As of 2026-08-29 19:06 (Asia/Singapore), the repo is primarily written in Triton, but pivoting some test cases to CUTLASS / CuTe DSL / CUDA C++ is acceptable if Triton itself is a blocker or bottleneck — particularly applicable to multi-CTA ops for which Triton support is currently lacking.

Since hardware counters are not available on the H200 node, use `triton.testing.Benchmark`. First figure out which is binding: compute or bandwidth. Then use Triton benchmarking to obtain percentage of theoretical max of that metric achieved by the kernel. Use this figure to guide optimization.

Bit-exactness should not be a priority if the kernel is already under the numerical gate.
