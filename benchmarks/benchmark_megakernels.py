#!/usr/bin/env python3
"""Compact latency benchmark for all currently supported megakernel cases."""

from __future__ import annotations

import argparse

import torch
import triton

from benchmarks.torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    generate_random_case,
)
from kernels.case8_attention import attention_tuning
from kernels.dag_megakernel import is_step_4_shape, resolved_dag_tuning
from kernels.fused_megakernel import resolved_megakernel_tuning
from optimized_transformer import IMPLEMENTED_CASES, make_optimized_transformer


def make_case(case_number: int):
    batch, sequence, model, heads = IMPLEMENTED_CASES[case_number]
    config = TransformerConfig(
        batch_size=batch,
        seq_len=sequence,
        d_model=model,
        num_heads=heads,
        ffn_dim=model,
        num_layers=4,
        causal=True,
    )
    torch.manual_seed(1234)
    baseline = BaselineTransformer(config).cuda().half().eval()
    optimized = make_optimized_transformer(baseline).cuda().eval()
    return optimized, config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="+",
        type=int,
        choices=IMPLEMENTED_CASES,
        default=IMPLEMENTED_CASES,
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rep", type=int, default=500)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("an NVIDIA GPU is required")
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (9, 0):
        raise RuntimeError(f"Hopper sm_90 is required, got {properties.name}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(
        f"torch={torch.__version__} triton={triton.__version__} "
        f"gpu={properties.name} sms={properties.multi_processor_count}"
    )

    with torch.inference_mode():
        for case_number in dict.fromkeys(args.cases):
            optimized, config = make_case(case_number)
            value, valid = generate_random_case(
                config,
                torch.device("cuda"),
                torch.float16,
                101234 + case_number,
                0.0,
                1.0,
            )
            # Populate the mask specialization and compile before timing.
            output = optimized(value, valid)

            def invoke() -> None:
                optimized(value, valid)

            median, low, high = triton.testing.do_bench(
                invoke,
                warmup=args.warmup,
                rep=args.rep,
                quantiles=[0.5, 0.2, 0.8],
            )
            batch, sequence, model, heads = IMPLEMENTED_CASES[case_number]
            if case_number == 8:
                family = "hybrid"
                tuning = attention_tuning()
            elif case_number == 13:
                family = "hybrid"
                tuning = {
                    "attention": "compiled-exact/cudnn-adaptive",
                    "linear_epilogues": "triton",
                }
            elif is_step_4_shape(value, heads):
                family = "dag"
                tuning = resolved_dag_tuning(batch, sequence)
            else:
                family = "resident"
                tuning = resolved_megakernel_tuning(batch, heads, model)
            print(
                f"case={case_number:2d} family={family:8s} "
                f"median={float(median):.6f} ms p20={float(low):.6f} "
                f"p80={float(high):.6f} checksum={output.float().sum().item():.8g} "
                f"tuning={tuning}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
