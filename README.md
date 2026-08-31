# TikTok TechJam 2026 submission: task 3

Task: optimize the absolute hell out of a simple transformer model.

## Model structure

We are given a standard transformer architecture, where each block is as follows:

```python
class BaselineTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = BaselineSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)
        x = x + self.ffn_out(F.gelu(self.ffn_in(self.norm2(x)), approximate="none"))

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x
```

where the full Transformer is simply a composition of these blocks, followed by normalization and masking. For the full architecture, see the [organizer-provided benchmark file](benchmarks/torch_transformer_benchmark.py).


## Approach

We aim to beat trivial optimizations like PyTorch SDPA and cuDNN attention kernels by implementing the full transformer (including all its constituent blocks) as a megakernel.

Production callers use one constructor for every implemented case:

```python
import copy

from optimized_transformer import make_optimized_transformer

optimized = make_optimized_transformer(copy.deepcopy(baseline)).eval()
output = optimized(value, valid_token_mask)
```

The supplied model must already contain CUDA FP16 parameters. The constructor
detects its benchmark case from `model.config`, prepares the appropriate
resident/DAG/hybrid implementation, and rejects unsupported shapes.

## Setup and build

The repository locks both kinds of dependencies used by the production path:

- Python packages are resolved exactly by `uv.lock`. In particular, the Linux
  environment uses PyTorch 2.13.0+cu129, Triton 3.7.1, and the complete cuDNN
  9.24.0.43 wheel.
- FlashAttention is a submodule at commit
  `ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820`. Its nested CUTLASS submodule is
  pinned at `7127592069c2fe01b041e174ba4345ef9b279671`.

Prerequisites are Git, `uv`, a C++ compiler, Ninja, and CUDA 12.9 installed at
`/usr/local/cuda-12.9`. The build deliberately fails if that exact CUDA
toolkit is unavailable. Production kernels require an NVIDIA Hopper GPU
(compute capability 9.0); compiling the FA3 wheel itself does not require a
GPU. If `uv` is not already installed, install it first with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone and build from a fresh checkout with:

```bash
git clone https://github.com/Scallion3008/tttj-2026.git
cd tttj-2026
./build.sh
```

`build.sh` initializes only the required submodules, creates `.venv` from the
frozen lockfile, temporarily applies the tracked CUDA 12.9 compatibility
patch, builds the production FA3 specialization, and installs it into `.venv`.
The vendor checkout is restored after the build and the wheel is saved under
`.artifacts/wheels/`. The build is idempotent. `MAX_JOBS` and `NVCC_THREADS`
may be set to tune build parallelism; their defaults are 4 and 2 respectively.

On the SoC Slurm cluster, submit the CPU-only build from the repository root:

```bash
sbatch job-scripts/build_fa3.sh
```

The production FA3 build is forward-only SM90 FP16 with head-dim 64 and split
support. Cases using head-dim 32 pad to that specialization. Backward, SM80,
FP8, and unused head dimensions are omitted to keep compilation bounded.

The only production vendor patch is
`patches/flash-attention/system-cuda-12.9.patch`. It makes FA3 use the required
system CUDA 12.9 compiler instead of downloading another nvcc. The experimental
score-rounding FA3 patch documented in `llm-scratchpad/step_8_results.md` was
slower and was rejected, so it is intentionally not part of the build.

### Running

Activate the environment and put the pinned cuDNN wheel ahead of any system
cuDNN installation:

```bash
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="${CUDA_HOME}/bin:${PATH}"
export CUDNN_ROOT="${PWD}/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="${CUDNN_ROOT}:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_MODULE_LOADING=LAZY
```

Then run the full correctness suite on an H200-141 allocation:

```bash
sbatch --gres=gpu:h200-141:1 job-scripts/validate_all_cases_h200.sh
```

For a local interactive Hopper allocation, the equivalent entrypoint is:

```bash
uv run --frozen python -m benchmarks.regression --cases 1 2 3 4 5 6 7 8 9 10 11 12 13 14
```

The CUDA extensions under `kernels/csrc/` are compiled just in time by PyTorch
on first use and cached in its normal extension cache; they need no separate
build step.

## Repository layout

- `optimized_transformer.py`: public optimized-model constructor and case map.
- `kernels/`: Triton kernels, optimized model adapters, and CUDA sources.
- `benchmarks/`: organizer reference plus correctness and latency benchmarks.
- `profiling/`: profiling, numerical validation, and diagnostic entrypoints.
- `job-scripts/`: Slurm scripts only.
- `job-scripts/outputs/`: Slurm logs and Nsight report artifacts.
- `llm-scratchpad/`: optimization notes and recorded results.
- `vendor/`: pinned Git submodules, including FlashAttention and its CUTLASS
  dependency.
- `patches/`: versioned patches applied to vendor sources by `build.sh`.



## Correctness evaluation

**Hard numerical gate vs PyTorch:** absolute error < 0.001, relative error < 0.01.

Numerical correctness utilities are located in [benchmarks/torch_transformer_benchmark.py](benchmarks/torch_transformer_benchmark.py).


### Performance evaluation

The performance of our implementation will be evaluated on a fixed set of input shapes:

No. | Batch Size | QKV Dim | Heads | Seq Len | Layers | Causal | FFN Dim
--- | ---------- | ------- | ----- | ------- | ------ | ------ | -------
1   | 64         | 128     | 4     | 128     | 4      | TRUE   | 128
2   | 1          | 128     | 4     | 128     | 4      | TRUE   | 128
3   | 4          | 128     | 4     | 128     | 4      | TRUE   | 128
4   | 16         | 128     | 4     | 128     | 4      | TRUE   | 128
5   | 128        | 128     | 4     | 128     | 4      | TRUE   | 128
6   | 10000      | 128     | 4     | 128     | 4      | TRUE   | 128
7   | 64         | 32      | 4     | 128     | 4      | TRUE   | 32
8   | 64         | 1024    | 4     | 128     | 4      | TRUE   | 1024
9   | 64         | 128     | 1     | 128     | 4      | TRUE   | 128
10  | 64         | 128     | 2     | 128     | 4      | TRUE   | 128
11  | 64         | 128     | 16    | 128     | 4      | TRUE   | 128
12  | 64         | 128     | 4     | 32      | 4      | TRUE   | 128
13  | 64         | 128     | 4     | 1024    | 4      | TRUE   | 128
14  | 32         | 1024    | 16    | 100000  | 2      | TRUE   | 1024
