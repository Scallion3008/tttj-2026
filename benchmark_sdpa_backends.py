#!/usr/bin/env python3
"""Measure every available PyTorch SDPA backend for the supported cases."""

from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext

import torch
import triton
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmark_step_4 import SDPATransformer
from profile_megakernel import CASES
from torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    generate_random_case,
)


BACKENDS = {
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", type=int, choices=CASES, default=CASES)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=300)
    parser.add_argument("--allow-h200", action="store_true")
    args = parser.parse_args()

    properties = torch.cuda.get_device_properties(0)
    allowed_name = "H100" in properties.name or (
        args.allow_h200 and "H200" in properties.name
    )
    if (properties.major, properties.minor) != (9, 0) or not allowed_name:
        raise RuntimeError(f"an H100/H200 sm_90 GPU is required, got {properties.name}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(
        f"torch={torch.__version__} triton={triton.__version__} "
        f"gpu={properties.name} cudnn={torch.backends.cudnn.version()}"
    )

    with torch.inference_mode():
        for case_number in dict.fromkeys(args.cases):
            batch, sequence, model_dimension, heads = CASES[case_number]
            config = TransformerConfig(
                batch_size=batch,
                seq_len=sequence,
                d_model=model_dimension,
                num_heads=heads,
                ffn_dim=model_dimension,
                num_layers=4,
                causal=True,
            )
            torch.manual_seed(1234)
            baseline = BaselineTransformer(config).cuda().half().eval()
            model = SDPATransformer(copy.deepcopy(baseline)).cuda().eval()
            value, valid = generate_random_case(
                config,
                torch.device("cuda"),
                torch.float16,
                200000 + case_number,
                0.0,
                1.0,
            )
            results: dict[str, float] = {}
            # xgpk0's H200 image lacks cuDNN's runtime-compiled engines; even
            # probing auto/cuDNN can terminate the process instead of raising.
            backend_names = (
                ("flash", "efficient", "math")
                if "H200" in properties.name
                else ("auto", *BACKENDS)
            )
            for name in backend_names:
                context = nullcontext() if name == "auto" else sdpa_kernel(BACKENDS[name])
                try:
                    with context:
                        model(value, valid)
                        median, low, high = triton.testing.do_bench(
                            lambda: model(value, valid),
                            warmup=args.warmup,
                            rep=args.rep,
                            quantiles=[0.5, 0.2, 0.8],
                        )
                    results[name] = float(median)
                    print(
                        f"case={case_number:2d} backend={name:9s} "
                        f"median={float(median):.6f} ms p20={float(low):.6f} "
                        f"p80={float(high):.6f}"
                    )
                except RuntimeError as error:
                    print(f"case={case_number:2d} backend={name:9s} unavailable: {error}")
            fastest = min(results, key=results.get)
            print(
                f"case={case_number:2d} fastest={fastest} "
                f"median={results[fastest]:.6f} ms"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
