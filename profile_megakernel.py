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

from benchmark_steps_1_2 import make_models
from sequence_resident import SequenceResidentTransformer
from torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    generate_random_case,
)


CASES = {
    1: (64, 128, 4),
    2: (1, 128, 4),
    3: (4, 128, 4),
    4: (16, 128, 4),
    5: (128, 128, 4),
    6: (10000, 128, 4),
    9: (64, 128, 1),
    10: (64, 128, 2),
    12: (64, 32, 4),
}


def make_case(case_number: int):
    batch, sequence, heads = CASES[case_number]
    if sequence == 128:
        _, optimized, config = make_models(
            batch, 1234, False, num_heads=heads
        )
        return optimized, config

    config = TransformerConfig(
        batch_size=batch,
        seq_len=sequence,
        d_model=128,
        num_heads=heads,
        ffn_dim=128,
        num_layers=4,
        causal=True,
    )
    torch.manual_seed(1234)
    baseline = BaselineTransformer(config).cuda().half().eval()
    optimized = SequenceResidentTransformer(copy.deepcopy(baseline)).cuda().eval()
    optimized.prepare()
    return optimized, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=CASES, required=True)
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
