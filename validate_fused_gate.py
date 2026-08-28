#!/usr/bin/env python3
"""Stress the production fused path with masked and repeated inputs."""

from __future__ import annotations

import os

import torch

from benchmark_steps_1_2 import make_models
from torch_transformer_benchmark import compare_outputs, generate_random_case


def main() -> int:
    batch_size = int(os.environ.get("VALIDATE_BATCH", "128"))
    trials = int(os.environ.get("VALIDATE_TRIALS", "5"))
    padding_values = tuple(
        float(value)
        for value in os.environ.get("VALIDATE_PADDING", "0.0,0.25,0.75").split(",")
    )
    baseline, optimized, config = make_models(batch_size, 1234, False)
    passed = True
    with torch.inference_mode():
        for padding_ratio in padding_values:
            for trial in range(trials):
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
                if not result.passed:
                    absolute = (reference - candidate).abs()
                    allowed = torch.maximum(
                        torch.full_like(absolute, 0.001),
                        reference.abs() * 0.01,
                    )
                    locations = torch.nonzero(absolute > allowed)[:8]
                    for location in locations:
                        key = tuple(location.tolist())
                        print(
                            f"  {key}: reference={reference[key].item():.8g} "
                            f"candidate={candidate[key].item():.8g} "
                            f"abs={absolute[key].item():.8g} "
                            f"allowed={allowed[key].item():.8g}"
                        )

                # The fused path must also be deterministic across successive
                # executions of the exact same launch.
                repeated = optimized(value, valid_mask)
                exact = bool(torch.equal(candidate, repeated))
                passed &= exact
                repeat_differences = int((candidate != repeated).sum().item())
                print(
                    f"  repeated launch: {'EXACT' if exact else 'DIFF'} "
                    f"elements={repeat_differences}"
                )
                if not exact:
                    repeat_locations = torch.nonzero(candidate != repeated)[:8]
                    for location in repeat_locations:
                        key = tuple(location.tolist())
                        print(
                            f"    {key}: first={candidate[key].item():.8g} "
                            f"second={repeated[key].item():.8g}"
                        )

    print(f"fused gate stress: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
