#!/usr/bin/env python3
"""Run one warmed transformer megakernel launch for Nsight Compute.

The profiler wrapper filters on the Triton kernel name, so this process may
freely use Torch kernels while constructing the model and input.  ``--case``
covers every production shape currently dispatched to either megakernel
family.
"""

from __future__ import annotations

import argparse
import copy

import torch

from benchmarks.torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    generate_random_case,
)
from optimized_transformer import (
    IMPLEMENTED_CASES,
    IMPLEMENTED_CASE_LAYERS,
    make_optimized_transformer,
)


def make_case(case_number: int):
    batch, sequence, model, heads = IMPLEMENTED_CASES[case_number]
    config = TransformerConfig(
        batch_size=batch,
        seq_len=sequence,
        d_model=model,
        num_heads=heads,
        ffn_dim=model,
        num_layers=IMPLEMENTED_CASE_LAYERS[case_number],
        causal=True,
    )
    torch.manual_seed(1234)
    baseline = BaselineTransformer(config).cuda().half().eval()
    optimized = make_optimized_transformer(copy.deepcopy(baseline)).cuda().eval()
    return optimized, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", type=int, choices=IMPLEMENTED_CASES, required=True
    )
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    optimized, config = make_case(args.case)
    value, valid_mask = generate_random_case(
        config=config,
        device=torch.device("cuda"),
        dtype=torch.float16,
        seed=101234,
        padding_ratio=0.0,
        input_scale=1.0,
    )
    with torch.inference_mode():
        for _ in range(args.warmup):
            optimized(value, valid_mask)
        torch.cuda.synchronize()
        output = optimized(value, valid_mask)
        torch.cuda.synchronize()
    print(
        f"profile launch complete: case={args.case}, "
        f"shape={(config.batch_size, config.seq_len, config.d_model)}, "
        f"heads={config.num_heads}, "
        f"checksum={output.float().sum().item():.8g}"
    )


if __name__ == "__main__":
    main()
