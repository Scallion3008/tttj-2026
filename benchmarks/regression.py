#!/usr/bin/env python3
"""Cross-case correctness regression for the public optimized-model factory."""

from __future__ import annotations

import argparse
import copy
import gc

import torch

from benchmarks.torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    compare_outputs,
    generate_random_case,
)
from optimized_transformer import (
    IMPLEMENTED_CASES,
    IMPLEMENTED_CASE_LAYERS,
    make_optimized_transformer,
)


RTOL = 0.01
ATOL = 0.001
SEED = 1234


def make_models(case_number: int):
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
    torch.manual_seed(SEED)
    baseline = BaselineTransformer(config).cuda().half().eval()
    optimized = make_optimized_transformer(copy.deepcopy(baseline)).cuda().eval()
    return baseline, optimized, config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="+",
        type=int,
        choices=IMPLEMENTED_CASES,
        default=tuple(IMPLEMENTED_CASES),
    )
    parser.add_argument(
        "--padding-ratios",
        nargs="+",
        type=float,
        default=(0.0, 0.25),
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--case14-reference-query-chunk", type=int, default=128)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("an NVIDIA GPU is required")
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (9, 0):
        raise RuntimeError(f"Hopper sm_90 is required, got {properties.name}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(
        f"torch={torch.__version__} gpu={properties.name} "
        f"sms={properties.multi_processor_count}"
    )

    all_passed = True
    for case_number in dict.fromkeys(args.cases):
        baseline, optimized, config = make_models(case_number)
        with torch.inference_mode():
            for padding_ratio in args.padding_ratios:
                failed = 0
                repeat_diff = 0
                max_abs = 0.0
                total = 0
                for trial in range(args.trials):
                    runtime_config = config
                    if case_number == 14:
                        # A single batch still exercises the fixed 100k-token
                        # case while keeping the exact reference tractable.
                        runtime_config = TransformerConfig(
                            batch_size=1,
                            seq_len=config.seq_len,
                            d_model=config.d_model,
                            num_heads=config.num_heads,
                            ffn_dim=config.ffn_dim,
                            num_layers=config.num_layers,
                            causal=config.causal,
                        )
                    value, valid = generate_random_case(
                        runtime_config,
                        torch.device("cuda"),
                        torch.float16,
                        SEED
                        + 100_000 * case_number
                        + 1_000 * trial
                        + round(100 * padding_ratio),
                        padding_ratio,
                        1.0,
                    )
                    if case_number == 14:
                        from benchmarks.benchmark_step_8 import _chunked_reference

                        reference = _chunked_reference(
                            baseline,
                            value,
                            valid,
                            args.case14_reference_query_chunk,
                        )
                    else:
                        reference = baseline(value, valid)
                    candidate = optimized(value, valid).clone()
                    repeated = optimized(value, valid)
                    result = compare_outputs(
                        reference,
                        candidate,
                        2 * RTOL if case_number == 14 else RTOL,
                        2 * ATOL if case_number == 14 else ATOL,
                    )
                    failed += result.failed_elements
                    total += result.total_elements
                    max_abs = max(max_abs, result.max_abs_error)
                    repeat_diff += int((candidate != repeated).sum().item())
                passed = failed == 0 and repeat_diff == 0
                all_passed &= passed
                print(
                    f"case={case_number:2d} padding={padding_ratio:.2f} "
                    f"{'PASS' if passed else 'FAIL'} failed={failed}/{total} "
                    f"max_abs={max_abs:.7g} repeat_diff={repeat_diff}"
                )
        del baseline, optimized
        gc.collect()
        torch.cuda.empty_cache()

    print(f"regression {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
