#!/usr/bin/env python3
"""Single-call and component profiling entrypoint for case 14."""

from __future__ import annotations

import argparse
import os
import statistics

import torch
import torch.nn.functional as F

from kernels.case8_fusions import residual_layer_norm
from kernels.case14_softmax import exact_softmax, exact_softmax_stats


def _measure(callable_, repetitions: int) -> float:
    with torch.inference_mode():
        for _ in range(2):
            callable_()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
        for start, end in zip(starts, ends):
            start.record()
            callable_()
            end.record()
        torch.cuda.synchronize()
    return statistics.median(
        start.elapsed_time(end) for start, end in zip(starts, ends)
    )


def attention(batch: int, sequence: int, ncu: bool) -> None:
    from flash_attn_3.flash_attn_interface import flash_attn_qkvpacked_func

    qkv = torch.zeros(
        batch,
        sequence,
        3,
        16,
        64,
        device="cuda",
        dtype=torch.float16,
    )
    with torch.inference_mode():
        flash_attn_qkvpacked_func(qkv, causal=True)
        torch.cuda.synchronize()
        if ncu:
            torch.cuda.cudart().cudaProfilerStart()
            flash_attn_qkvpacked_func(qkv, causal=True)
            torch.cuda.cudart().cudaProfilerStop()
            torch.cuda.synchronize()
        else:
            elapsed = _measure(
                lambda: flash_attn_qkvpacked_func(qkv, causal=True), 3
            )
            flops = 2.0 * batch * sequence * (sequence + 1) * 1024
            print(
                f"attention_ms={elapsed:.6f} "
                f"useful_tflops={flops / (elapsed * 1e9):.3f}"
            )


def softmax(rows: int, sequence: int, ncu: bool) -> None:
    if rows > sequence:
        raise ValueError("softmax profiling rows must not exceed sequence length")
    query_start = sequence - rows
    fast_exp = bool(int(os.environ.get("TTTJ_STEP8_FAST_EXP", "0")))
    source = torch.randn(rows, sequence, device="cuda", dtype=torch.float16)
    value = source.clone()
    with torch.inference_mode():
        exact_softmax(
            value,
            sequence,
            query_start=query_start,
            input_scale=0.125,
            inplace=True,
            fast_exp=fast_exp,
        )
        torch.cuda.synchronize()
        if ncu:
            value.copy_(source)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            exact_softmax(
                value,
                sequence,
                query_start=query_start,
                input_scale=0.125,
                inplace=True,
                fast_exp=fast_exp,
            )
            torch.cuda.cudart().cudaProfilerStop()
            torch.cuda.synchronize()
        else:
            elapsed = _measure(
                lambda: exact_softmax(
                    source,
                    sequence,
                    query_start=query_start,
                    input_scale=0.125,
                    inplace=False,
                    fast_exp=fast_exp,
                ),
                3,
            )
            valid_elements = rows * (2 * sequence - rows + 1) // 2
            # Three score reads plus one probability write. The causal suffix
            # is not read, but is written as zero in the in-place epilogue.
            effective_bytes = 6 * valid_elements + 2 * rows * sequence
            print(
                f"softmax_ms={elapsed:.6f} "
                f"effective_tbps={effective_bytes / (elapsed * 1e9):.3f}"
            )


def exact_tile(batch: int, sequence: int, rows: int) -> None:
    query_start = sequence - rows
    fast_exp = bool(int(os.environ.get("TTTJ_STEP8_FAST_EXP", "0")))
    query = torch.randn(
        batch, 16, rows, 64, device="cuda", dtype=torch.float16
    )
    key = torch.randn(
        batch, 16, sequence, 64, device="cuda", dtype=torch.float16
    )
    value = torch.randn_like(key)
    qk_ms = _measure(lambda: torch.matmul(query, key.transpose(-2, -1)), 3)
    scores = torch.matmul(query, key.transpose(-2, -1))
    softmax_ms = _measure(
        lambda: exact_softmax(
            scores,
            sequence,
            query_start=query_start,
            input_scale=0.125,
            inplace=False,
            fast_exp=fast_exp,
        ),
        3,
    )
    probabilities = exact_softmax(
        scores,
        sequence,
        query_start=query_start,
        input_scale=0.125,
        inplace=False,
        fast_exp=fast_exp,
    )
    pv_ms = _measure(lambda: torch.matmul(probabilities, value), 3)
    gemm_flops = 2.0 * batch * 16 * rows * sequence * 64
    query_bytes = batch * 16 * rows * 64 * 2
    key_value_bytes = batch * 16 * sequence * 64 * 2
    score_bytes = batch * 16 * rows * sequence * 2
    qk_bytes = query_bytes + key_value_bytes + score_bytes
    pv_bytes = score_bytes + key_value_bytes + query_bytes
    valid_elements = batch * 16 * rows * (2 * sequence - rows + 1) // 2
    stored_elements = batch * 16 * rows * sequence
    effective_bytes = 6 * valid_elements + 2 * stored_elements
    print(f"exact_tile batch={batch} rows={rows} sequence={sequence}")
    print(
        f"qk_ms={qk_ms:.6f} tensor_tflops={gemm_flops / (qk_ms * 1e9):.3f} "
        f"minimum_tbps={qk_bytes / (qk_ms * 1e9):.3f}"
    )
    print(
        f"softmax_ms={softmax_ms:.6f} "
        f"effective_tbps={effective_bytes / (softmax_ms * 1e9):.3f}"
    )
    print(
        f"pv_ms={pv_ms:.6f} tensor_tflops={gemm_flops / (pv_ms * 1e9):.3f} "
        f"minimum_tbps={pv_bytes / (pv_ms * 1e9):.3f}"
    )


def fused_tile(batch: int, sequence: int, rows: int) -> None:
    """Tune the exact-statistics plus fused probability/PV tile."""
    from kernels.case14_attention import score_value

    query_start = sequence - rows
    query = torch.randn(
        batch, 16, rows, 64, device="cuda", dtype=torch.float16
    )
    key = torch.randn(
        batch, 16, sequence, 64, device="cuda", dtype=torch.float16
    )
    value = torch.randn_like(key)
    scores = torch.matmul(query, key.transpose(-2, -1))
    qk_ms = _measure(lambda: torch.matmul(query, key.transpose(-2, -1)), 3)
    statistics = exact_softmax_stats(
        scores,
        sequence,
        query_start=query_start,
        input_scale=0.125,
    )
    stats_ms = _measure(
        lambda: exact_softmax_stats(
            scores,
            sequence,
            query_start=query_start,
            input_scale=0.125,
        ),
        3,
    )
    print(f"fused_tile batch={batch} rows={rows} sequence={sequence}")
    valid_elements = batch * 16 * rows * (2 * sequence - rows + 1) // 2
    stats_bytes = 4 * valid_elements
    qk_flops = 2.0 * batch * 16 * rows * sequence * 64
    qk_bytes = (
        batch * 16 * rows * 64 * 2
        + batch * 16 * sequence * 64 * 2
        + batch * 16 * rows * sequence * 2
    )
    print(
        f"qk_ms={qk_ms:.6f} tensor_tflops={qk_flops / (qk_ms * 1e9):.3f} "
        f"minimum_tbps={qk_bytes / (qk_ms * 1e9):.3f}"
    )
    print(
        f"stats_ms={stats_ms:.6f} "
        f"effective_tbps={stats_bytes / (stats_ms * 1e9):.3f}"
    )
    gemm_flops = 2.0 * batch * 16 * rows * sequence * 64
    variants = (
        (32, 64, 4, 3),
        (32, 128, 4, 3),
        (32, 128, 8, 3),
        (64, 64, 4, 3),
        (64, 128, 4, 3),
        (64, 128, 8, 2),
        (64, 128, 8, 3),
        (64, 128, 8, 4),
        (64, 256, 8, 3),
        (128, 32, 8, 3),
        (128, 64, 4, 3),
        (128, 64, 8, 2),
        (128, 64, 8, 3),
        (128, 64, 8, 4),
        (128, 128, 8, 2),
        (128, 128, 8, 3),
        (256, 32, 8, 3),
        (256, 64, 8, 3),
    )
    for pv_rows, pv_keys, warps, stages in variants:
        try:
            elapsed = _measure(
                lambda: score_value(
                    scores,
                    statistics,
                    value,
                    query_start,
                    rows=pv_rows,
                    keys=pv_keys,
                    num_warps=warps,
                    num_stages=stages,
                ),
                3,
            )
            score_bytes = batch * 16 * rows * sequence * 2
            value_bytes = (
                batch
                * 16
                * ((rows + pv_rows - 1) // pv_rows)
                * sequence
                * 64
                * 2
            )
            output_bytes = batch * 16 * rows * 64 * 2
            minimum_bytes = score_bytes + value_bytes + output_bytes
            print(
                f"pv_rows={pv_rows} pv_keys={pv_keys} warps={warps} "
                f"stages={stages} fused_pv_ms={elapsed:.6f} "
                f"tensor_tflops={gemm_flops / (elapsed * 1e9):.3f} "
                f"minimum_tbps={minimum_bytes / (elapsed * 1e9):.3f}"
            )
        except Exception as error:
            print(
                f"pv_rows={pv_rows} pv_keys={pv_keys} warps={warps} "
                f"stages={stages} error={error!r}"
            )
    fast_elapsed = _measure(
        lambda: score_value(
            scores,
            statistics,
            value,
            query_start,
            rows=128,
            keys=64,
            num_warps=8,
            num_stages=3,
            fast_exp=True,
        ),
        3,
    )
    print(
        f"pv_rows=128 pv_keys=64 warps=8 stages=3 fast_exp=1 "
        f"fused_pv_ms={fast_elapsed:.6f} "
        f"tensor_tflops={gemm_flops / (fast_elapsed * 1e9):.3f}"
    )


def components(rows: int) -> None:
    value = torch.randn(rows, 1024, device="cuda", dtype=torch.float16)
    branch = torch.randn_like(value)
    weight = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    qkv_weight = torch.randn(3072, 1024, device="cuda", dtype=torch.float16)
    bias = torch.randn(1024, device="cuda", dtype=torch.float16)
    qkv_bias = torch.randn(3072, device="cuda", dtype=torch.float16)
    norm_weight = torch.randn(1024, device="cuda", dtype=torch.float16)
    norm_bias = torch.randn(1024, device="cuda", dtype=torch.float16)
    valid = torch.ones(rows, device="cuda", dtype=torch.bool)

    tests = {
        "linear": lambda: F.linear(value, weight, bias),
        "packed_qkv": lambda: F.linear(value, qkv_weight, qkv_bias),
        "layer_norm": lambda: F.layer_norm(value, (1024,), norm_weight, norm_bias),
        "gelu": lambda: F.gelu(value, approximate="none"),
        "residual_norm": lambda: residual_layer_norm(
            value,
            branch,
            norm_weight,
            norm_bias,
            valid,
            all_valid=True,
        ),
    }
    for name, callable_ in tests.items():
        elapsed = _measure(callable_, 5)
        if name == "linear":
            useful = 2.0 * rows * 1024 * 1024 / (elapsed * 1e9)
            metric = f"tflops={useful:.3f}"
        elif name == "packed_qkv":
            useful = 2.0 * rows * 1024 * 3072 / (elapsed * 1e9)
            metric = f"tflops={useful:.3f}"
        else:
            metric = ""
        print(f"{name}_ms={elapsed:.6f} {metric}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("attention", "softmax", "exact-tile", "fused-tile", "components"),
        required=True,
    )
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--sequence", type=int, default=100_000)
    parser.add_argument("--rows", type=int, default=3_200_000)
    parser.add_argument("--ncu", action="store_true")
    args = parser.parse_args()
    properties = torch.cuda.get_device_properties(0)
    print(f"gpu={properties.name} sm={properties.multi_processor_count}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.mode == "attention":
        attention(args.batch, args.sequence, args.ncu)
    elif args.mode == "softmax":
        softmax(args.rows, args.sequence, args.ncu)
    elif args.mode == "exact-tile":
        exact_tile(args.batch, args.sequence, args.rows)
    elif args.mode == "fused-tile":
        fused_tile(args.batch, args.sequence, args.rows)
    else:
        components(args.rows)


if __name__ == "__main__":
    main()
