#!/usr/bin/env python3
"""Compare every production kernel with whole-model torch.compile timing."""

from __future__ import annotations

import argparse
import copy
import gc
from dataclasses import dataclass
from typing import Optional

import torch
import triton

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


SEED = 1234
RTOL = 0.01
ATOL = 0.001


@dataclass(frozen=True)
class Measurement:
    milliseconds: float
    low: float
    high: float
    passed: Optional[bool]
    failed: Optional[int]
    max_abs: Optional[float]


def make_config(case_number: int) -> TransformerConfig:
    batch, sequence, model, heads = IMPLEMENTED_CASES[case_number]
    return TransformerConfig(
        batch_size=batch,
        seq_len=sequence,
        d_model=model,
        num_heads=heads,
        ffn_dim=model,
        num_layers=IMPLEMENTED_CASE_LAYERS[case_number],
        causal=True,
    )


def measure(
    model: torch.nn.Module,
    value: torch.Tensor,
    valid: torch.Tensor,
    reference: torch.Tensor,
    warmup: int,
    rep: int,
    *,
    validate: bool = True,
) -> Measurement:
    with torch.inference_mode():
        # The first call performs Inductor compilation, autotuning, or custom
        # Triton compilation. It is intentionally outside the timing window.
        candidate = model(value, valid).clone()
        result = (
            compare_outputs(reference, candidate, RTOL, ATOL)
            if validate
            else None
        )
        median, low, high = triton.testing.do_bench(
            lambda: model(value, valid),
            warmup=warmup,
            rep=rep,
            quantiles=[0.5, 0.2, 0.8],
        )
    return Measurement(
        float(median),
        float(low),
        float(high),
        result.passed if result is not None else None,
        result.failed_elements if result is not None else None,
        result.max_abs_error if result is not None else None,
    )


def print_measurement(
    case_number: int,
    provider: str,
    measurement: Measurement,
    optimized_ms: float,
) -> None:
    relative = measurement.milliseconds / optimized_ms
    accuracy = (
        f"{'PASS' if measurement.passed else 'FAIL'} "
        f"failed={measurement.failed} max_abs={measurement.max_abs:.7g}"
        if measurement.passed is not None
        else "accuracy=UNCHECKED"
    )
    print(
        f"case={case_number:2d} provider={provider:24s} "
        f"median={measurement.milliseconds:.6f} ms "
        f"p20={measurement.low:.6f} p80={measurement.high:.6f} "
        f"provider_over_production={relative:.3f}x "
        f"{accuracy}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="+",
        type=int,
        choices=IMPLEMENTED_CASES,
        default=IMPLEMENTED_CASES,
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("default", "reduce-overhead", "max-autotune"),
        default=("default", "reduce-overhead", "max-autotune"),
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--long-context-warmup", type=int, default=3_000)
    parser.add_argument("--long-context-rep", type=int, default=15_000)
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

    for case_number in dict.fromkeys(args.cases):
        config = make_config(case_number)
        torch.manual_seed(SEED)
        baseline = BaselineTransformer(config).cuda().half().eval()
        optimized = make_optimized_transformer(
            copy.deepcopy(baseline)
        ).cuda().eval()
        value, valid = generate_random_case(
            config,
            torch.device("cuda"),
            torch.float16,
            SEED + 100_000 + case_number,
            0.0,
            1.0,
        )
        if case_number == 14:
            # The organizer implementation materializes [B,H,S,S] scores:
            # 32*16*100000^2 FP16 values are about 9.3 TiB.  Production uses
            # streaming attention, so measure it and report the intrinsically
            # infeasible eager/compiled providers explicitly.
            optimized_result = measure(
                optimized,
                value,
                valid,
                reference=value,
                warmup=args.long_context_warmup,
                rep=args.long_context_rep,
                validate=False,
            )
            print(
                "case=14 provider=torch-eager              SKIP "
                "reason=materialized_attention_requires_9.3_TiB"
            )
            print_measurement(
                case_number,
                "production",
                optimized_result,
                optimized_result.milliseconds,
            )
            for mode in dict.fromkeys(args.modes):
                print(
                    f"case=14 provider=torch-compile-{mode:12s} SKIP "
                    "reason=materialized_attention_requires_9.3_TiB"
                )
            del baseline, optimized, value, valid
            gc.collect()
            torch.cuda.empty_cache()
            continue
        with torch.inference_mode():
            reference = baseline(value, valid).clone()

        eager = measure(
            baseline, value, valid, reference, args.warmup, args.rep
        )
        optimized_result = measure(
            optimized, value, valid, reference, args.warmup, args.rep
        )
        print_measurement(
            case_number, "torch-eager", eager, optimized_result.milliseconds
        )
        print_measurement(
            case_number, "production", optimized_result,
            optimized_result.milliseconds
        )

        for mode in dict.fromkeys(args.modes):
            try:
                # Each case/mode is a distinct specialization of the same
                # Python ``forward`` code object. Without resetting Dynamo,
                # its per-code recompile limit is reached partway through the
                # sweep and later rows cease to be valid compiled references.
                torch.compiler.reset()
                compiled = torch.compile(
                    copy.deepcopy(baseline),
                    fullgraph=True,
                    dynamic=False,
                    mode=mode,
                )
                compiled_result = measure(
                    compiled,
                    value,
                    valid,
                    reference,
                    args.warmup,
                    args.rep,
                    validate=False,
                )
                print_measurement(
                    case_number,
                    f"torch-compile-{mode}",
                    compiled_result,
                    optimized_result.milliseconds,
                )
                del compiled
            except Exception as error:
                print(
                    f"case={case_number:2d} provider=torch-compile-{mode:12s} "
                    f"ERROR {type(error).__name__}: {error}"
                )
            gc.collect()
            torch.cuda.empty_cache()

        del baseline, optimized, value, valid, reference
        gc.collect()
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
