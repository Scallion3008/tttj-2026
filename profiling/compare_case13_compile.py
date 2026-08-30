"""Repeated paired H100 timing for production case 13 and torch.compile."""

from __future__ import annotations

import copy
import statistics

import torch
import triton

from benchmarks.torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    compare_outputs,
    generate_random_case,
)
from kernels.case13_hybrid import (
    CASE13_BATCH,
    CASE13_HEADS,
    CASE13_LAYERS,
    CASE13_MODEL,
    CASE13_SEQUENCE,
    Case13OptimizedTransformer,
    GraphedCase13Hybrid,
)


def main() -> int:
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (9, 0):
        raise RuntimeError(f"Hopper sm_90 is required, got {properties.name}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = TransformerConfig(
        CASE13_BATCH,
        CASE13_SEQUENCE,
        CASE13_MODEL,
        CASE13_HEADS,
        CASE13_MODEL,
        CASE13_LAYERS,
        True,
    )
    torch.manual_seed(1234)
    baseline = BaselineTransformer(config).cuda().half().eval()
    production = GraphedCase13Hybrid(
        Case13OptimizedTransformer(copy.deepcopy(baseline))
    ).cuda().eval()
    torch.compiler.reset()
    compiled = torch.compile(
        copy.deepcopy(baseline),
        fullgraph=True,
        dynamic=False,
        mode="max-autotune",
    )
    value, valid = generate_random_case(
        config,
        torch.device("cuda"),
        torch.float16,
        101_247,
        0.0,
        1.0,
    )

    with torch.inference_mode():
        reference = baseline(value, valid)
        production.prepare(value, valid)
        actual = production(value, valid)
        accuracy = compare_outputs(reference, actual, 0.01, 0.001)
        compiled(value, valid)
        print(
            f"gpu={properties.name} sms={properties.multi_processor_count} "
            f"production_accuracy={'PASS' if accuracy.passed else 'FAIL'} "
            f"failed={accuracy.failed_elements} max_abs={accuracy.max_abs_error:.7g}"
        )

        timings: dict[str, list[float]] = {"production": [], "compile": []}
        for round_index in range(6):
            order = (
                ("production", production),
                ("compile", compiled),
            )
            if round_index % 2:
                order = tuple(reversed(order))
            for name, model in order:
                median, low, high = triton.testing.do_bench(
                    lambda model=model: model(value, valid),
                    warmup=100,
                    rep=500,
                    quantiles=[0.5, 0.2, 0.8],
                )
                timings[name].append(float(median))
                print(
                    f"round={round_index} provider={name:10s} "
                    f"median={float(median):.6f} ms "
                    f"p20={float(low):.6f} p80={float(high):.6f}"
                )

        production_median = statistics.median(timings["production"])
        compile_median = statistics.median(timings["compile"])
        print(
            f"paired production={production_median:.6f} ms "
            f"compile={compile_median:.6f} ms "
            f"compile_over_production={compile_median / production_median:.4f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
