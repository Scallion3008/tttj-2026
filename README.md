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

To reproduce our speedups, run the following or its `sbatch` equivalent on an
H200-141 allocation:

```bash
uv run --frozen python -m benchmarks.benchmark_compile_comparison
```

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


## Results

Geomean speedup: **5.76x** vs. eager, **2.34x** vs. `torch.compile`.

| Case | Production | Eager | Compile | Eager / production | Compile / production |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.169280 ms | 0.888320 ms | 0.355008 ms | 5.248x | 2.097x |
| 2 | 0.114272 ms | 0.768128 ms | 0.356992 ms | 6.722x | 3.124x |
| 3 | 0.109408 ms | 0.776000 ms | 0.357280 ms | 7.093x | 3.266x |
| 4 | 0.109664 ms | 0.778496 ms | 0.355552 ms | 7.099x | 3.242x |
| 5 | 0.262880 ms | 1.389344 ms | 0.517344 ms | 5.285x | 1.968x |
| 6 | 12.374400 ms | 85.928352 ms | 26.261248 ms | 6.944x | 2.122x |
| 7 | 0.095456 ms | 0.809440 ms | 0.354976 ms | 8.480x | 3.719x |
| 8 | 1.065472 ms | 2.323616 ms | 1.292960 ms | 2.181x | 1.214x |
| 9 | 0.158224 ms | 0.690816 ms | 0.291424 ms | 4.366x | 1.842x |
| 10 | 0.155680 ms | 0.786304 ms | 0.358176 ms | 5.051x | 2.301x |
| 11 | 0.220128 ms | 1.881632 ms | 0.619712 ms | 8.548x | 2.815x |
| 12 | 0.104352 ms | 0.766224 ms | 0.353232 ms | 7.343x | 3.385x |
| 13 | 4.443712 ms | 20.356400 ms | 5.360304 ms | 4.581x | 1.206x |
| 14 | 2927.799072 ms | OOM | OOM | N/A | N/A |


## Limitations

### Megakernels are hyper-specialized

By their nature, megakernels are extremely specialized to the hardware they run on, as well as the model architecture for which they are built. Our insights from the development process are generalizable, but the kernels themselves are not.

### Triton is not sufficiently fine-grained for maximum performance

While suitably high-level for convenience, Triton ultimately loses to CUDA C++ in optimization potential. In particular, Triton lacks support for some advanced Hopper hardware features such as DSMEM. If the competition were longer, we would seriously have considered rewriting all kernels in CUDA C++ or CuTe DSL.


## If we had more time, we would...

- Implement all kernels in a lower-level language like CuTe DSL, Gluon, or CUDA C++, making full use of Hopper hardware features.
- Spend more time tuning numerics, particularly those of optimized libraries. If Flash Attention 3, cuDNN Frontend or Torch SDPA could be made to pass the numerical tolerance gate, they would be usable in more layers of our hybrid strategies, likely giving significant speedup.
