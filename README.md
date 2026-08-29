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

## Repository layout

- `optimized_transformer.py`: public optimized-model constructor and case map.
- `kernels/`: Triton kernels, optimized model adapters, and CUDA sources.
- `benchmarks/`: organizer reference plus correctness and latency benchmarks.
- `profiling/`: profiling, numerical validation, and diagnostic entrypoints.
- `job-scripts/`: Slurm scripts only.
- `job-scripts/outputs/`: Slurm logs and Nsight report artifacts.
- `llm-scratchpad/`: optimization notes and recorded results.



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
