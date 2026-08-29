#!/usr/bin/env python3
"""Compare cases 5/6 with auto-selected PyTorch SDPA on Hopper."""

from __future__ import annotations

import argparse
import copy

import torch
import triton

from benchmark_step_4 import SDPATransformer
from benchmark_steps_1_2 import make_models
from torch_transformer_benchmark import compare_outputs, generate_random_case


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", nargs="+", type=int, choices=(128, 10000), default=(128, 10000))
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rep", type=int, default=1000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("an NVIDIA GPU is required")
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (9, 0) or "H100" not in properties.name:
        raise RuntimeError(f"an H100 sm_90 GPU is required, got {properties.name}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(
        f"torch={torch.__version__} triton={triton.__version__} "
        f"gpu={properties.name} cudnn={torch.backends.cudnn.version()}"
    )

    with torch.inference_mode():
        for batch_size in dict.fromkeys(args.batch_sizes):
            baseline, optimized, config = make_models(batch_size, 1234, False)
            sdpa = SDPATransformer(copy.deepcopy(baseline)).cuda().eval()
            value, valid = generate_random_case(
                config,
                torch.device("cuda"),
                torch.float16,
                101234 + batch_size,
                0.0,
                1.0,
            )
            reference = baseline(value, valid)
            outputs = {
                "megakernel": optimized(value, valid),
                "sdpa-auto": sdpa(value, valid),
            }
            for provider, output in outputs.items():
                result = compare_outputs(reference, output, rtol=0.01, atol=0.001)
                print(
                    f"B={batch_size:5d} {provider:10s} accuracy="
                    f"{'PASS' if result.passed else 'FAIL'} failed="
                    f"{result.failed_elements}/{result.total_elements} "
                    f"max_abs={result.max_abs_error:.7g}"
                )

            providers = {
                "megakernel": optimized,
                "sdpa-auto": sdpa,
                "torch": baseline,
            }
            times: dict[str, float] = {}
            for provider, model in providers.items():
                median, low, high = triton.testing.do_bench(
                    lambda model=model: model(value, valid),
                    warmup=args.warmup,
                    rep=args.rep,
                    quantiles=[0.5, 0.2, 0.8],
                )
                times[provider] = float(median)
                print(
                    f"B={batch_size:5d} {provider:10s} median={float(median):.6f} ms "
                    f"p20={float(low):.6f} p80={float(high):.6f}"
                )
            print(
                f"B={batch_size:5d} speedup_vs_sdpa="
                f"{times['sdpa-auto'] / times['megakernel']:.3f}x "
                f"speedup_vs_torch={times['torch'] / times['megakernel']:.3f}x"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
