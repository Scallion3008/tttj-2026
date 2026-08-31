#!/usr/bin/env python3
"""Correctness and latency harness for step 8 / benchmark case 14."""

from __future__ import annotations

import argparse
import copy
import statistics

import torch
import torch.nn.functional as F

from benchmarks.torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    compare_outputs,
    generate_random_case,
)
from kernels.case14_hybrid import Case14LayerwiseHybrid


FULL_CONFIG = TransformerConfig(
    batch_size=32,
    seq_len=100_000,
    d_model=1024,
    num_heads=16,
    ffn_dim=1024,
    num_layers=2,
    causal=True,
)


def _chunked_exact_attention(
    attention,
    value: torch.Tensor,
    valid_token_mask: torch.Tensor,
    query_chunk: int,
) -> torch.Tensor:
    """Memory-bounded reference preserving the benchmark FP16 boundaries."""
    batch, sequence, _ = value.shape

    def split_heads(projected: torch.Tensor) -> torch.Tensor:
        return projected.view(batch, sequence, 16, 64).transpose(1, 2).contiguous()

    q = split_heads(attention.q_proj(value))
    k = split_heads(attention.k_proj(value))
    v = split_heads(attention.v_proj(value))
    output = torch.empty_like(q)
    key_positions = torch.arange(sequence, device=value.device)
    invalid_keys = ~valid_token_mask[:, None, None, :]
    for query_start in range(0, sequence, query_chunk):
        query_end = min(query_start + query_chunk, sequence)
        scores = torch.matmul(
            q[:, :, query_start:query_end], k.transpose(-2, -1)
        ) * (64**-0.5)
        query_positions = torch.arange(query_start, query_end, device=value.device)
        causal = key_positions[None, :] > query_positions[:, None]
        scores = scores.masked_fill(causal, float("-inf"))
        scores = scores.masked_fill(invalid_keys, float("-inf"))
        probabilities = torch.softmax(scores.float(), dim=-1).to(value.dtype)
        output[:, :, query_start:query_end] = torch.matmul(probabilities, v)
    context = output.transpose(1, 2).contiguous().view(batch, sequence, 1024)
    branch = attention.out_proj(context)
    return branch.masked_fill(~valid_token_mask[..., None], 0)


def _chunked_reference(
    model: BaselineTransformer,
    value: torch.Tensor,
    valid_token_mask: torch.Tensor,
    query_chunk: int,
) -> torch.Tensor:
    x = value
    with torch.inference_mode():
        for layer in model.layers:
            x = x + _chunked_exact_attention(
                layer.attention,
                layer.norm1(x),
                valid_token_mask,
                query_chunk,
            )
            x = x + layer.ffn_out(
                F.gelu(layer.ffn_in(layer.norm2(x)), approximate="none")
            )
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        x = model.final_norm(x)
        return x.masked_fill(~valid_token_mask[..., None], 0)


def _measure(callable_, warmup: int, repetitions: int) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmup):
            callable_()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
        for start, end in zip(starts, ends):
            start.record()
            callable_()
            end.record()
        torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def _make_models(provider: str, pack_qkv: bool, fuse_norm: bool):
    torch.manual_seed(1234)
    baseline = BaselineTransformer(FULL_CONFIG).cuda().half().eval()
    if provider == "production":
        from optimized_transformer import make_optimized_transformer

        candidate = make_optimized_transformer(copy.deepcopy(baseline)).eval()
    else:
        candidate = Case14LayerwiseHybrid(
            copy.deepcopy(baseline),
            attention_backend=provider,
            pack_qkv=pack_qkv,
            fuse_residual_norm=fuse_norm,
        ).eval()
    candidate.prepare()
    return baseline, candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=(
            "production",
            "fa3",
            "flash",
            "cudnn",
            "triton",
            "blend",
            "exact",
            "exact-fused",
            "exact-first",
            "exact-fused-first",
            "exact-fused-first-cudnn",
            "exact-fused-first-flash",
            "exact-fused-first-fa3-cudnn",
            "exact-fused-first-fa3-flash",
            "exact-fused-h4-first",
            "exact-fused-h8-first",
            "exact-fused-h12-first",
            "exact-last",
            "cudnn-first",
            "flash-first",
            "triton-first",
            "blend-first",
            "fa3-cudnn-first",
            "fa3-split2-first",
            "fa3-split4-first",
            "fa3-split8-first",
            "fa3-split2",
            "fa3-split4",
            "fa3-split8",
        ),
        default="production",
    )
    parser.add_argument("--no-pack-qkv", action="store_true")
    parser.add_argument("--no-fuse-norm", action="store_true")
    parser.add_argument("--accuracy-sequence", type=int, default=2048)
    parser.add_argument("--accuracy-batch", type=int, default=1)
    parser.add_argument("--accuracy-trials", type=int, default=1)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--input-scales", nargs="+", type=float, default=(1.0,))
    parser.add_argument("--full-sequence-accuracy", action="store_true")
    parser.add_argument("--reference-query-chunk", type=int, default=128)
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--skip-performance", action="store_true")
    parser.add_argument("--performance-batch", type=int, default=32)
    parser.add_argument("--performance-sequence", type=int, default=100_000)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    properties = torch.cuda.get_device_properties(0)
    print(
        f"gpu={properties.name} sm={properties.multi_processor_count} "
        f"memory={properties.total_memory / 2**30:.2f} GiB "
        f"torch={torch.__version__} cudnn={torch.backends.cudnn.version()}"
    )
    production = args.provider == "production"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    baseline, candidate = _make_models(
        args.provider, not args.no_pack_qkv, not args.no_fuse_norm
    )
    print(
        f"provider={args.provider} "
        f"pack_qkv={candidate.pack_qkv if production else not args.no_pack_qkv} "
        f"fuse_norm={True if production else not args.no_fuse_norm}"
    )

    if not args.skip_accuracy:
        print("accuracy:")
        for scale_index, input_scale in enumerate(args.input_scales):
            for trial in range(args.accuracy_trials):
                accuracy_sequence = (
                    100_000
                    if args.full_sequence_accuracy
                    else args.accuracy_sequence
                )
                config = TransformerConfig(
                    batch_size=args.accuracy_batch,
                    seq_len=accuracy_sequence,
                    d_model=1024,
                    num_heads=16,
                    ffn_dim=1024,
                    num_layers=2,
                    causal=True,
                )
                value, mask = generate_random_case(
                    config,
                    torch.device("cuda"),
                    torch.float16,
                    4321 + 1000 * scale_index + trial,
                    args.padding_ratio,
                    input_scale,
                )
                with torch.inference_mode():
                    reference = (
                        _chunked_reference(
                            baseline, value, mask, args.reference_query_chunk
                        )
                        if args.full_sequence_accuracy
                        else baseline(value, mask)
                    )
                    output = candidate(value, mask)
                result = compare_outputs(
                    reference, output, rtol=args.rtol, atol=args.atol
                )
                route = ""
                if production and candidate._last_input_rms is not None:
                    route = (
                        f" rms={candidate._last_input_rms:.8g}"
                        f" attention={candidate._active_attention_backend}"
                    )
                print(
                    f"  scale={input_scale:.8g} trial={trial} "
                    f"{route} "
                    f"passed={result.passed} "
                    f"failed={result.failed_elements}/{result.total_elements} "
                    f"max_abs={result.max_abs_error:.8g} "
                    f"mean_abs={result.mean_abs_error:.8g}"
                )
                if not result.passed:
                    difference = (output.float() - reference.float()).abs()
                    failed = (difference > args.atol) & (
                        difference > args.rtol * reference.float().abs()
                    )
                    indices = torch.nonzero(failed, as_tuple=False)[:32]
                    details = []
                    for index in indices.tolist():
                        location = tuple(index)
                        details.append(
                            (
                                location,
                                float(reference[location]),
                                float(output[location]),
                                float(difference[location]),
                            )
                        )
                    print(f"  failures(index, reference, output, abs)={details}")
                del value, mask, reference, output

    if not args.skip_performance:
        shape = (args.performance_batch, args.performance_sequence, 1024)
        value = torch.randn(shape, device="cuda", dtype=torch.float16)
        mask = torch.ones(shape[:2], device="cuda", dtype=torch.bool)
        torch.cuda.reset_peak_memory_stats()
        samples = _measure(
            lambda: candidate(value, mask), args.warmup, args.repetitions
        )
        median_ms = statistics.median(samples)
        batch, sequence = shape[:2]
        attention_flops = (
            2.0 * 2 * batch * sequence * (sequence + 1) * 1024
        )
        linear_flops = 2.0 * 12 * batch * sequence * 1024 * 1024
        useful_flops = attention_flops + linear_flops
        print(
            f"performance shape={shape} samples_ms={samples} "
            f"median_ms={median_ms:.6f} "
            f"useful_tflops={useful_flops / (median_ms * 1e9):.3f}"
        )
        if production and candidate._last_input_rms is not None:
            print(
                f"performance_rms={candidate._last_input_rms:.8g} "
                f"attention={candidate._active_attention_backend}"
            )
        print(f"peak_allocated_gib={torch.cuda.max_memory_allocated() / 2**30:.3f}")


if __name__ == "__main__":
    main()
