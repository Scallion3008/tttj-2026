"""Find numerically safe per-layer Inductor-softmax masks for case 13."""

from __future__ import annotations

import argparse
import copy

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
    Case13LayerwiseHybrid,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--scales", nargs="+", type=float, default=(0.5, 0.75, 1.0, 2.0, 3.0)
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--kind", choices=("softmax", "full"), default="softmax")
    args = parser.parse_args()

    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (9, 0):
        raise RuntimeError(f"Hopper sm_90 is required, got {properties.name}")
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
    candidate = Case13LayerwiseHybrid(
        copy.deepcopy(baseline),
        pack_qkv=True,
        head_major_qkv_projection=True,
        attention_backend="cudnn",
        fuse_linear_epilogues=True,
        fuse_input_norm=True,
        streaming_attention_mask=0,
        compile_exact_attention=True,
    ).cuda().eval()
    candidate.prepare()
    failures = [0] * (1 << CASE13_LAYERS)
    worst = [0.0] * (1 << CASE13_LAYERS)
    scale_failures = {
        scale: [0] * (1 << CASE13_LAYERS) for scale in args.scales
    }

    print(
        f"gpu={properties.name} sms={properties.multi_processor_count} "
        f"kind={args.kind} trials={args.trials} scales={args.scales}"
    )
    with torch.inference_mode():
        for scale in args.scales:
            for trial in range(args.trials):
                value, valid = generate_random_case(
                    config,
                    torch.device("cuda"),
                    torch.float16,
                    100_000 + trial + round(scale * 10_000),
                    0.0,
                    scale,
                )
                reference = baseline(value, valid)
                for mask in range(1 << CASE13_LAYERS):
                    if args.kind == "softmax":
                        candidate.compiled_softmax_mask = mask
                    else:
                        candidate.compiled_full_attention_mask = mask
                    actual = candidate(value, valid)
                    result = compare_outputs(reference, actual, 0.01, 0.001)
                    failures[mask] += result.failed_elements
                    scale_failures[scale][mask] += result.failed_elements
                    worst[mask] = max(worst[mask], result.max_abs_error)

        for mask in range(1 << CASE13_LAYERS):
            print(
                f"mask=0x{mask:x} layers={mask.bit_count()} "
                f"failed={failures[mask]} max_abs={worst[mask]:.7g}"
            )
        for scale in args.scales:
            print(
                f"scale={scale:g} failures="
                + ",".join(
                    f"0x{mask:x}:{failed}"
                    for mask, failed in enumerate(scale_failures[scale])
                )
            )

        if args.skip_timing:
            return 0

        value, valid = generate_random_case(
            config,
            torch.device("cuda"),
            torch.float16,
            201_234,
            0.0,
            1.0,
        )
        for mask in range(1 << CASE13_LAYERS):
            if failures[mask] != 0:
                continue
            if args.kind == "softmax":
                candidate.compiled_softmax_mask = mask
            else:
                candidate.compiled_full_attention_mask = mask
            candidate(value, valid)
            median, low, high = triton.testing.do_bench(
                lambda: candidate(value, valid),
                warmup=args.warmup,
                rep=args.rep,
                quantiles=[0.5, 0.2, 0.8],
            )
            print(
                f"timing mask=0x{mask:x} median={float(median):.6f} ms "
                f"p20={float(low):.6f} p80={float(high):.6f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
