#!/usr/bin/env python3
"""Stress the production fused path with masked and repeated inputs."""

from __future__ import annotations

import torch

from benchmark_steps_1_2 import make_models
from torch_transformer_benchmark import compare_outputs, generate_random_case


def main() -> int:
    baseline, optimized, config = make_models(128, 1234, False)
    passed = True
    with torch.inference_mode():
        for padding_ratio in (0.0, 0.25, 0.75):
            for trial in range(5):
                value, valid_mask = generate_random_case(
                    config=config,
                    device=torch.device("cuda"),
                    dtype=torch.float16,
                    seed=4000 + trial + round(100 * padding_ratio),
                    padding_ratio=padding_ratio,
                    input_scale=1.0,
                )
                reference = baseline(value, valid_mask)
                candidate = optimized(value, valid_mask)
                result = compare_outputs(reference, candidate, rtol=0.01, atol=0.001)
                passed &= result.passed
                print(
                    f"padding={padding_ratio:.2f} trial={trial} "
                    f"{'PASS' if result.passed else 'FAIL'} "
                    f"failed={result.failed_elements}/{result.total_elements} "
                    f"max_abs={result.max_abs_error:.7g}"
                )

                # A barrier-free compile must also be deterministic across
                # successive executions of the exact same launch.
                repeated = optimized(value, valid_mask)
                exact = bool(torch.equal(candidate, repeated))
                passed &= exact
                print(f"  repeated launch: {'EXACT' if exact else 'DIFF'}")

    print(f"fused gate stress: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
