#!/usr/bin/env python3
"""Run one warmed transformer megakernel launch for Nsight Compute."""

from __future__ import annotations

import argparse

import torch

from benchmark_steps_1_2 import make_models
from torch_transformer_benchmark import generate_random_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, choices=(128, 10000), required=True)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, optimized, config = make_models(args.batch_size, 1234, False)
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
        f"profile launch complete: batch={args.batch_size}, "
        f"checksum={output.float().sum().item():.8g}"
    )


if __name__ == "__main__":
    main()
