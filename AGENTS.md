# Instructions

This repo seeks to optimize the forward pass of a simple transformer model by implementing it as a collection of megakernels. Refer to [README.md](README.md) for details, including numerical accuracy constraints and implementation details.

## Environment

If the hostname begins with `xlogin` or `xcnc`, you are working on a Slurm login node. Submit jobs to run on H200-141 GPUs using `sbatch --gres=gpu:h200-141:1 <script path>`.

H200 compute node `xgpk0` should have CUDA installed at `/usr/local/cuda-12.9/`. If CUDA 12.9 is not available, fail immediately instead of falling back to another CUDA version.

## Repo structure

- `optimized_transformer.py`: single construction API for all implemented cases.
- `kernels`: optimized Triton kernels, model adapters, and CUDA sources.
- `benchmarks`: correctness and performance benchmarks, including the reference.
- `profiling`: profiling, validation, and diagnostic entrypoints.
- `llm-scratchpad`: notes left by previous agent sessions for consumption by other agents.
- `job-scripts`: Slurm job scripts only.
- `job-scripts/outputs`: Slurm logs and profiler output artifacts.

## Optimization methodology

First work out the rough kernel structure (plans for each test case are detailed in `llm-scratchpad/initial_notes.md`), then hunt for potential algorithmic optimizations, and iterate until max possible speedup is achieved. As of 2026-08-29 19:06 (Asia/Singapore), the repo is primarily written in Triton, but pivoting some test cases to CUTLASS / CuTe DSL / CUDA C++ is acceptable if Triton itself is a blocker or bottleneck — particularly applicable to multi-CTA ops for which Triton support is currently lacking.

Bit-exactness should not be a priority if the kernel is already under the numerical gate.

### Profiling

Profiling should be run with Nsight Compute (located at `/usr/local/cuda-12.9/bin/ncu`) on H100-96 nodes (use `--gres=gpu:h100-96:1`). If H100-96 nodes are unavailable, use `triton.testing.Benchmark` on the H200-141 node. Note that hardware counters are disabled on the latter, and therefore `ncu` will fail unless only source-based metrics (and no runtime ones) are collected.

For each kernel being profiled, first figure out which is binding: compute or bandwidth. Then use `ncu` or Triton benchmarking to obtain percentage of theoretical max of that metric achieved by the kernel. Use this figure to guide optimization, along with other metrics reported by `ncu` if available.
